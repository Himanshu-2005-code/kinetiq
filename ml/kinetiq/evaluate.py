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


# ----------------------------------------------------------------------------
# Sessions with breaks: mirrors the browser engine (web/src/engine.ts)

ABSENT_AFTER = 8      # frames without a visible body before counting holds
SWITCH_AFTER = 5      # consecutive confident inferences before auto mode switches
SWITCH_CONF = 0.7


def make_break_session(rng: np.random.Generator, exercise: str, gap: int, length: int = 300):
    """Two sets of one exercise with the person out of frame for `gap` frames in between."""
    a = sample_clip(rng, length, exercise=exercise, fps=30.0, idle_margin=False)
    b = sample_clip(rng, length, exercise=exercise, fps=30.0, idle_margin=False)
    away = np.repeat(a.landmarks[-1:], gap, 0).copy()
    away[..., 2] = 0.0  # tracker reports nothing visible
    seq = np.concatenate([a.landmarks, away, b.landmarks])
    present = np.concatenate([np.ones(length), np.zeros(gap), np.ones(length)]).astype(bool)
    truth = sum(count_reps(np.stack([np.cos(c.phase), np.sin(c.phase)], -1), c.active) for c in (a, b))
    return seq, present, truth


@torch.no_grad()
def stream_session(model, seq: np.ndarray, present: np.ndarray, window: int = 64,
                   stride: int = 4) -> dict[str, int]:
    """Count reps frame by frame with the same rules as the live coach (auto mode).

    * Counting holds while nobody is in frame; the window restarts on return.
    * The first prediction on a fresh window replays all of its frames, so reps
      finished during the warm-up still count.
    * Auto mode only switches exercise after a sustained confident detection,
      and never resets a count.
    """
    model.eval()
    counters = {e: RepCounter() for e in EXERCISES[1:]}
    buf, since, absent, fresh = [], 0, 0, True
    ema = np.zeros(len(EXERCISES))
    current = candidate = None
    streak = 0
    for t in range(len(seq)):
        absent = 0 if present[t] else absent + 1
        if absent >= ABSENT_AFTER:
            for c in counters.values():
                c.update((1.0, 0.0), 0.0)
            buf, since, fresh = [], 0, True
            continue
        buf.append(seq[t])
        buf = buf[-window:]
        since += 1
        if len(buf) < window or since < stride:
            continue
        out = model(torch.from_numpy(np.stack(buf)[None]))
        probs = out["exercise"].softmax(-1)[0].numpy()
        ema = probs if fresh else 0.7 * ema + 0.3 * probs
        detected = EXERCISES[int(ema.argmax())]
        if detected != "idle" and ema.max() > SWITCH_CONF and detected != current:
            streak = streak + 1 if detected == candidate else 1
            candidate = detected
            if streak >= SWITCH_AFTER or current is None:
                current, candidate, streak = detected, None, 0
        elif detected == current:
            candidate, streak = None, 0
        phase = torch.nn.functional.normalize(out["phase"], dim=-1)[0].numpy()
        active = torch.sigmoid(out["active"])[0].numpy()
        gate = current is not None and (detected == current or ema[0] < 0.6)
        n = window if fresh else since
        if current is not None:
            for k in range(window - n, window):
                counters[current].update((float(phase[k, 0]), float(phase[k, 1])),
                                         float(active[k]) if gate else 0.0)
        since, fresh = 0, False
    return {e: c.count for e, c in counters.items()}


def break_session_eval(model, n_sessions: int = 16, seed: int = 99) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    errs = []
    for k in range(n_sessions):
        ex = EXERCISES[1 + k % (len(EXERCISES) - 1)]
        seq, present, truth = make_break_session(rng, ex, gap=[30, 60, 150][k % 3])
        errs.append(stream_session(model, seq, present)[ex] - truth)
    errs = np.array(errs)
    return {"break_rep_mae": float(np.abs(errs).mean()), "break_rep_exact": float((errs == 0).mean())}


if __name__ == "__main__":
    import argparse
    import json

    from .export import load

    ap = argparse.ArgumentParser(description="Evaluate a checkpoint, including sessions with breaks")
    ap.add_argument("checkpoint")
    ap.add_argument("--sessions", type=int, default=16)
    args = ap.parse_args()
    m, _ = load(args.checkpoint)
    res = {**rep_counting_eval(m), **break_session_eval(m, args.sessions)}
    print(json.dumps({k: round(v, 4) for k, v in res.items()}, indent=1))
