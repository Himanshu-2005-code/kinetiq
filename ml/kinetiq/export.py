"""Export a trained checkpoint to ONNX for onnxruntime-web and check parity.

    python -m kinetiq.export runs/default/best.pt ../web/public/models/kinetiq.onnx
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .model import InferenceWrapper, KinetiqModel, ModelConfig
from .skeleton import EXERCISES, FAULTS, JOINTS, MEDIAPIPE_INDICES
from .synth import sample_batch

OUTPUTS = ["exercise_probs", "fault_probs", "phase", "active_probs"]


def load(ckpt_path: str) -> tuple[KinetiqModel, dict]:
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = KinetiqModel(ModelConfig(**ckpt["config"]["model"]))
    model.load_state_dict(ckpt["model"])
    return model.eval(), ckpt


def export(ckpt_path: str, out_path: str, opset: int = 17) -> dict:
    model, ckpt = load(ckpt_path)
    n_frames = ckpt["config"]["data"]["n_frames"]
    wrapper = InferenceWrapper(model).eval()
    dummy = torch.from_numpy(sample_batch(np.random.default_rng(0), 1, n_frames)["landmarks"])
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(wrapper, (dummy,), str(out_path), input_names=["landmarks"],
                      output_names=OUTPUTS, opset_version=opset, dynamo=False,
                      dynamic_axes={"landmarks": {0: "batch"}, **{o: {0: "batch"} for o in OUTPUTS}})

    # Parity check against PyTorch on fresh clips.
    import onnxruntime as ort
    sess = ort.InferenceSession(str(out_path), providers=["CPUExecutionProvider"])
    x = sample_batch(np.random.default_rng(1), 8, n_frames)["landmarks"]
    with torch.no_grad():
        ref = [t.numpy() for t in wrapper(torch.from_numpy(x))]
    got = sess.run(OUTPUTS, {"landmarks": x})
    max_err = max(float(np.abs(a - b).max()) for a, b in zip(ref, got))
    assert max_err < 1e-4, f"ONNX/PyTorch mismatch: {max_err}"

    meta = {"window": n_frames, "joints": JOINTS, "mediapipe_indices": MEDIAPIPE_INDICES,
            "exercises": EXERCISES, "faults": FAULTS, "outputs": OUTPUTS,
            "metrics": {k: round(float(v), 4) for k, v in ckpt.get("metrics", {}).items()},
            "parity_max_abs_err": max_err}
    json.dump(meta, open(out_path.with_suffix(".json"), "w"), indent=1)
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.2f} MB), parity err {max_err:.2e}")
    return meta


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("output")
    a = ap.parse_args()
    export(a.checkpoint, a.output)
