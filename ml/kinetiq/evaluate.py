"""Evaluation: clip-level metrics plus streaming rep counting on long sessions."""
from __future__ import annotations

import numpy as np
import torch

from .reps import RepCounter
from .skeleton import EXERCISES, FAULTS
from .synth import sample_clip


@torch.no_grad()
def clip_metrics(model, batch: dict[str, np.ndarray], device="cpu") -> dict[str, float]:
    model.eval()
    x = torch.from_numpy(batch["landmarks"]).to(device)
    out = model(x)
    ex = torch.from_numpy(batch["exercise"]).to(device)
    faults = torch.from_numpy(batch["faults"]).to(device)
    active = torch.from_numpy(batch["active"]).to(device)
    phase = torch.from_numpy(batch["phase"]).to(device)

    pred_ex = out["exercise"].argmax(-1)
    metrics = {"exercise_acc": (pred_ex == ex).float().mean().item()}
    for i, name in enumerate(EXERCISES):
        m = ex == i
        if m.any():
            metrics[f"acc/{name}"] = (pred_ex[m] == i).float().mean().item()

    fp = torch.sigmoid(out["faults"]) > 0.5
    for i, name in enumerate(FAULTS):
        t, p = faults[:, i] > 0.5, fp[:, i]
        tp = (t & p).sum().item()
        prec = tp / max(p.sum().item(), 1)
        rec = tp / max(t.sum().item(), 1)
        metrics[f"f1/{name}"] = 2 * prec * rec / max(prec + rec, 1e-9)
    metrics["fault_f1_macro"] = float(np.mean([metrics[f"f1/{n}"] for n in FAULTS]))

    pa = torch.atan2(out["phase"][..., 1], out["phase"][..., 0])
    err = torch.remainder(pa - phase + np.pi, 2 * np.pi) - np.pi
    mask = active > 0.5
    metrics["phase_mae_deg"] = (err.abs()[mask].mean() * 180 / np.pi).item() if mask.any() else 0.0
    metrics["active_acc"] = ((torch.sigmoid(out["active"]) > 0.5) == mask).float().mean().item()
    return metrics


@torch.no_grad()
def stream_predict(model, landmarks: np.ndarray, window: int = 64, stride: int = 4,
                   device="cpu") -> tuple[np.ndarray, np.ndarray]:
    """Run the model over a long sequence the way the browser does.

    Returns per-frame (phase cos/sin, active prob). Every frame's prediction
    comes from the most recent window that contains it, so there is no
    lookahead beyond `stride` frames.
    """
    model.eval()
    n = landmarks.shape[0]
    starts = list(range(0, max(n - window, 0) + 1, stride))
    if starts[-1] + window < n:
        starts.append(n - window)
    wins = np.stack([landmarks[s:s + window] for s in starts])
    out = model(torch.from_numpy(wins).to(device))
    ph = torch.nn.functional.normalize(out["phase"], dim=-1).cpu().numpy()
    act = torch.sigmoid(out["active"]).cpu().numpy()
    phase = np.zeros((n, 2), np.float32)
    active = np.zeros(n, np.float32)
    for w, s in enumerate(starts):
        lo = 0 if w == 0 else window - (s - starts[w - 1])
        phase[s + lo:s + window] = ph[w, lo:]
        active[s + lo:s + window] = act[w, lo:]
    return phase, active


def count_reps(phase: np.ndarray, active: np.ndarray) -> int:
    rc = RepCounter()
    for p, a in zip(phase, active):
        rc.update((float(p[0]), float(p[1])), float(a))
    return rc.count


def rep_counting_eval(model, n_sessions: int = 60, length: int = 360, seed: int = 1234,
                      device="cpu") -> dict[str, float]:
    rng = np.random.default_rng(seed)
    errs, per_ex = [], {e: [] for e in EXERCISES[1:]}
    for k in range(n_sessions):
        ex = EXERCISES[1 + k % (len(EXERCISES) - 1)]
        clip = sample_clip(rng, length, exercise=ex, fps=float(rng.uniform(20, 30)))
        gt_phase = np.stack([np.cos(clip.phase), np.sin(clip.phase)], -1)
        truth = count_reps(gt_phase, clip.active)
        ph, act = stream_predict(model, clip.landmarks, device=device)
        pred = count_reps(ph, act)
        errs.append(abs(pred - truth))
        per_ex[ex].append(abs(pred - truth))
    res = {"rep_mae": float(np.mean(errs)), "rep_exact": float(np.mean(np.array(errs) == 0))}
    for ex, e in per_ex.items():
        res[f"rep_mae/{ex}"] = float(np.mean(e))
    return res
