import math

import numpy as np
import pytest
import torch

from kinetiq.model import FEATURE_DIM, KinetiqModel, ModelConfig, PoseFeaturizer
from kinetiq.reps import RepCounter
from kinetiq.skeleton import BONES, EXERCISES, FAULTS, NUM_JOINTS
from kinetiq.synth import (_jumping_jack, _lunge, _pushup, _squat, sample_body, sample_batch,
                           sample_clip, two_bone_ik)
from kinetiq.train import compute_loss

SMALL = ModelConfig(d_model=32, n_layers=2, n_heads=2)


def test_batch_shapes_and_ranges():
    b = sample_batch(np.random.default_rng(0), 32, 48)
    assert b["landmarks"].shape == (32, 48, NUM_JOINTS, 3)
    assert b["faults"].shape == (32, len(FAULTS))
    assert np.isfinite(b["landmarks"]).all()
    assert (b["landmarks"][..., 2] >= 0).all() and (b["landmarks"][..., 2] <= 1).all()
    assert (b["phase"] >= 0).all() and (b["phase"] < 2 * math.pi + 1e-5).all()
    assert set(np.unique(b["exercise"])) <= set(range(len(EXERCISES)))


def test_idle_has_no_activity_or_faults():
    c = sample_clip(np.random.default_rng(1), 64, exercise="idle")
    assert c.active.sum() == 0 and c.faults.sum() == 0


def test_two_bone_ik_preserves_lengths():
    rng = np.random.default_rng(0)
    root = rng.normal(size=(100, 3))
    end = root + rng.normal(size=(100, 3)) * 0.3
    mid = two_bone_ik(root, end, 0.25, 0.22, np.array([0, 0, 1.0]))
    assert np.allclose(np.linalg.norm(mid - root, axis=-1), 0.25, atol=1e-6)


@pytest.mark.parametrize("fn", [_squat, _lunge, _jumping_jack, _pushup])
def test_bone_lengths_constant_over_time(fn):
    rng = np.random.default_rng(2)
    body = sample_body(rng)
    n = 60
    depth = (1 - np.cos(np.linspace(0, 4 * np.pi, n))) / 2
    world = fn(rng, n, depth, [], body, np.arange(n) / 30)
    for a, b in BONES:
        lengths = np.linalg.norm(world[:, a] - world[:, b], axis=-1)
        assert lengths.std() < 1e-3, (fn.__name__, a, b, lengths.std())


def test_featurizer_dim_and_mirror_invariant_angles():
    x = torch.from_numpy(sample_batch(np.random.default_rng(3), 4, 32)["landmarks"])
    f = PoseFeaturizer()(x)
    assert f.shape == (4, 32, FEATURE_DIM)
    mirrored = x.clone()
    mirrored[..., 0] = 2.0 - mirrored[..., 0]
    fm = PoseFeaturizer()(mirrored)
    # Angle, trunk and root features (everything after the coordinates/visibility) are mirror invariant.
    start = NUM_JOINTS * 2 * 2 + NUM_JOINTS
    assert torch.allclose(f[..., start:], fm[..., start:], atol=1e-4)


def test_featurizer_scale_and_translation_invariant():
    x = torch.from_numpy(sample_batch(np.random.default_rng(4), 2, 32)["landmarks"])
    y = x.clone()
    y[..., :2] = y[..., :2] * 1.7 + 0.3
    assert torch.allclose(PoseFeaturizer()(x), PoseFeaturizer()(y), atol=1e-4)


def test_model_forward_and_backward():
    model = KinetiqModel(SMALL)
    b = {k: torch.from_numpy(v) for k, v in sample_batch(np.random.default_rng(5), 8, 40).items()}
    out = model(b["landmarks"])
    assert out["exercise"].shape == (8, len(EXERCISES))
    assert out["phase"].shape == (8, 40, 2)
    loss, _ = compute_loss(out, b, {"exercise": 1, "faults": 1, "phase": 1, "active": 1,
                                    "label_smoothing": 0.0})
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_rep_counter_on_clean_phase():
    rc = RepCounter()
    frames = np.linspace(0, 5 * 2 * math.pi, 400)  # exactly five reps
    for a in frames:
        rc.update((math.cos(a), math.sin(a)), 1.0)
    assert rc.count == 5


def test_rep_counter_ignores_jitter_and_idle():
    rng = np.random.default_rng(0)
    rc = RepCounter()
    for a in np.linspace(0, 3 * 2 * math.pi, 300) + rng.normal(0, 0.15, 300):
        rc.update((math.cos(a), math.sin(a)), 1.0)
    for _ in range(50):  # standing still, model says inactive
        rc.update((1.0, 0.0), 0.1)
    assert rc.count == 3


def test_onnx_export_parity(tmp_path):
    pytest.importorskip("onnxruntime")
    from kinetiq.export import export
    model = KinetiqModel(SMALL)
    ckpt = tmp_path / "m.pt"
    torch.save({"model": model.state_dict(),
                "config": {"model": SMALL.__dict__, "data": {"n_frames": 64}}}, ckpt)
    meta = export(str(ckpt), str(tmp_path / "m.onnx"))
    assert meta["parity_max_abs_err"] < 1e-4
