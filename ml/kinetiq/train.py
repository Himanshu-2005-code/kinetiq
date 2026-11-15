"""Train the Kinetiq motion transformer on an endless stream of synthetic clips.

    python -m kinetiq.train --config configs/default.yaml
    python -m kinetiq.train --config configs/default.yaml --steps 200   # quick smoke run
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader, IterableDataset, get_worker_info

from .evaluate import clip_metrics, rep_counting_eval
from .model import KinetiqModel, ModelConfig
from .synth import sample_batch


class SyntheticStream(IterableDataset):
    """Infinite stream of freshly generated batches; each worker gets its own seed."""

    def __init__(self, batch_size: int, n_frames: int, seed: int):
        self.batch_size, self.n_frames, self.seed = batch_size, n_frames, seed

    def __iter__(self):
        info = get_worker_info()
        wid = info.id if info else 0
        rng = np.random.default_rng([self.seed, wid])
        while True:
            yield {k: torch.from_numpy(v) for k, v in sample_batch(rng, self.batch_size, self.n_frames).items()}


def compute_loss(out, batch, w: dict) -> tuple[torch.Tensor, dict]:
    l_ex = F.cross_entropy(out["exercise"], batch["exercise"], label_smoothing=w["label_smoothing"])
    l_fault = F.binary_cross_entropy_with_logits(out["faults"], batch["faults"])
    active = batch["active"]
    target = torch.stack([torch.cos(batch["phase"]), torch.sin(batch["phase"])], -1)
    cos_sim = F.cosine_similarity(out["phase"], target, dim=-1)
    l_phase = ((1 - cos_sim) * active).sum() / active.sum().clamp_min(1.0)
    l_active = F.binary_cross_entropy_with_logits(out["active"], active)
    total = (w["exercise"] * l_ex + w["faults"] * l_fault + w["phase"] * l_phase
             + w["active"] * l_active)
    return total, {"loss": total.item(), "l_ex": l_ex.item(), "l_fault": l_fault.item(),
                   "l_phase": l_phase.item(), "l_active": l_active.item()}


def lr_at(step: int, total: int, warmup: int, base: float, min_ratio: float = 0.05) -> float:
    if step < warmup:
        return base * (step + 1) / warmup
    p = (step - warmup) / max(total - warmup, 1)
    return base * (min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


@torch.no_grad()
def ema_update(ema: torch.nn.Module, model: torch.nn.Module, decay: float):
    for pe, pm in zip(ema.parameters(), model.parameters()):
        pe.lerp_(pm, 1 - decay)


def score(m: dict) -> float:
    """Single number for checkpoint selection."""
    return m["exercise_acc"] + m["fault_f1_macro"] + m["active_acc"] - m["phase_mae_deg"] / 90


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(open(args.config))
    tc = cfg["train"]
    if args.steps:
        tc["steps"] = args.steps
    out_dir = Path(args.out or cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(tc["seed"])
    torch.set_num_threads(tc.get("threads", torch.get_num_threads()))

    model = KinetiqModel(ModelConfig(**cfg["model"]))
    ema = copy.deepcopy(model).eval()
    for p in ema.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params / 1e6:.2f}M")

    decay, no_decay = [], []
    for n, p in model.named_parameters():
        (no_decay if p.ndim < 2 or n.endswith("cls") else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": tc["weight_decay"]},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=tc["lr"], betas=(0.9, 0.98))

    stream = SyntheticStream(tc["batch_size"], cfg["data"]["n_frames"], tc["seed"])
    loader = iter(DataLoader(stream, batch_size=None, num_workers=tc["workers"],
                             persistent_workers=tc["workers"] > 0))
    val = sample_batch(np.random.default_rng(10_000), cfg["data"]["val_size"], cfg["data"]["n_frames"])

    log, best, t0 = [], -1e9, time.time()
    for step in range(tc["steps"]):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step, tc["steps"], tc["warmup"], tc["lr"])
        batch = next(loader)
        out = model(batch["landmarks"])
        loss, parts = compute_loss(out, batch, cfg["loss"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tc["grad_clip"])
        opt.step()
        ema_update(ema, model, tc["ema_decay"] if step > tc["warmup"] else 0.0)

        if step % tc["log_every"] == 0:
            print(f"step {step:5d} lr {opt.param_groups[0]['lr']:.2e} "
                  + " ".join(f"{k} {v:.3f}" for k, v in parts.items())
                  + f" | {time.time() - t0:.0f}s", flush=True)
        if (step + 1) % tc["eval_every"] == 0 or step + 1 == tc["steps"]:
            m = clip_metrics(ema, val)
            m["step"] = step + 1
            log.append(m)
            print("  eval " + " ".join(f"{k}={v:.3f}" for k, v in m.items() if "/" not in k), flush=True)
            if score(m) > best:
                best = score(m)
                torch.save({"model": ema.state_dict(), "config": cfg, "metrics": m}, out_dir / "best.pt")

    ckpt = torch.load(out_dir / "best.pt", weights_only=False)
    ema.load_state_dict(ckpt["model"])
    final = {**clip_metrics(ema, val), **rep_counting_eval(ema)}
    print("final " + json.dumps({k: round(v, 4) for k, v in final.items()}, indent=1))
    json.dump({"history": log, "final": final, "params": n_params,
               "train_seconds": time.time() - t0}, open(out_dir / "metrics.json", "w"), indent=1)


if __name__ == "__main__":
    main()
