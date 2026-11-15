"""Streaming rep counter driven by the model's per-frame phase output.

The model predicts rep progress as an angle. A rep is one full turn, so
counting means unwrapping the angle over time and counting completed turns.
Hysteresis makes jitter harmless: backwards wobble lowers the running total
but can never un-count a rep, and a rep only counts once the total passes the
next full turn. web/src/reps.ts mirrors this logic line for line.
"""
from __future__ import annotations

import math

TWO_PI = 2 * math.pi


def wrap(a: float) -> float:
    """Wrap an angle to [-pi, pi)."""
    return (a + math.pi) % TWO_PI - math.pi


class RepCounter:
    def __init__(self, active_threshold: float = 0.5, max_step: float = math.pi * 0.9):
        self.active_threshold = active_threshold
        self.max_step = max_step  # ignore implausible jumps (tracking glitches)
        self.reset()

    def reset(self):
        self.count = 0
        self.total = 0.0
        self.last = None

    def update(self, cos_sin: tuple[float, float], active_prob: float) -> int:
        angle = math.atan2(cos_sin[1], cos_sin[0]) % TWO_PI
        if active_prob < self.active_threshold:
            self.last = None
            return self.count
        if self.last is None:
            # (Re)starting: anchor the total to where we are inside the current rep.
            start = angle - TWO_PI if angle > 1.5 * math.pi else angle
            self.total = self.count * TWO_PI + start
        else:
            step = wrap(angle - self.last)
            if abs(step) < self.max_step:
                self.total += step
        self.last = angle
        while self.total >= (self.count + 1) * TWO_PI:
            self.count += 1
        return self.count
