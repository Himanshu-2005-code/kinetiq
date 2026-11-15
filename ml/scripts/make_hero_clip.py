"""Write the looping squat clip shown on the landing page (web/public/hero-clip.json)."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kinetiq.synth import _phase_signal, _squat, sample_body  # noqa: E402

rng = np.random.default_rng(7)
fps, n = 30.0, 300  # five reps of two seconds each
body = sample_body(rng)
phase, depth, active = _phase_signal(np.random.default_rng(1), n, fps, 2.0, 0, 0)
phase = (np.arange(n) / (2.0 * fps)) % 1.0 * 2 * np.pi        # exact, seamless loop
depth = (1 - np.cos(phase)) / 2
world = _squat(np.random.default_rng(3), n, depth, [], body, np.arange(n) / fps)
yaw = 0.75  # three-quarter view shows both depth and knee travel
rot = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
cam = world @ rot.T
aspect = 880 / 748
scale = 0.86 / np.ptp(cam[..., 1])
x = cam[..., 0] * scale
y = -cam[..., 1] * scale
x = x - (x.min() + x.max()) / 2 + aspect * 0.42
y = y - y.max() + 0.93
frames = np.stack([x, y, np.ones_like(x)], -1).round(4)
out = Path(__file__).resolve().parents[2] / "web/public/hero-clip.json"
json.dump({"aspect": aspect, "fps": fps, "frames": frames.tolist(),
           "phase": phase.round(4).tolist(), "active": [1] * n}, open(out, "w"), separators=(",", ":"))
print("wrote", out, out.stat().st_size // 1024, "KB")
