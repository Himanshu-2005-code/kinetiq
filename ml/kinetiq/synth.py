"""Procedural motion synthesis: labelled pose sequences without a motion-capture dataset.

Each clip is produced in four stages:

1. Sample a body (segment lengths) and an exercise, plus any form faults.
2. Drive a 3D kinematic skeleton with a per-rep phase signal. Legs and arms
   are solved with two-bone IK, so bone lengths stay constant.
3. Render through a randomised weak-perspective camera (yaw, elevation, scale,
   framing, mirroring) into the same normalised image space the web client
   feeds the model: x is multiplied by the frame aspect ratio, y is in [0, 1].
4. Corrupt it the way a real pose estimator would: jitter, low-frequency drift,
   self-occlusion from depth, random joint dropouts and frame stalls.

World frame: +x is the person's left, +y is up, +z is the direction they face.
Units are fractions of body height.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .skeleton import APPLICABLE_FAULTS, EXERCISES, FAULTS, J, NUM_JOINTS

TWO_PI = 2.0 * np.pi


@dataclass
class Clip:
    landmarks: np.ndarray  # (T, 13, 3) x, y, visibility
    exercise: int
    faults: np.ndarray     # (len(FAULTS),) multi-hot
    phase: np.ndarray      # (T,) rep progress in [0, 2pi)
    active: np.ndarray     # (T,) 1 while an exercise rep is in progress


# --------------------------------------------------------------------------- body

def sample_body(rng: np.random.Generator) -> dict:
    return {
        "shoulder_w": rng.uniform(0.21, 0.26),
        "hip_w": rng.uniform(0.16, 0.20),
        "torso": rng.uniform(0.27, 0.32),
        "head": rng.uniform(0.11, 0.14),
        "upper_arm": rng.uniform(0.16, 0.20),
        "forearm": rng.uniform(0.14, 0.17),
        "thigh": rng.uniform(0.23, 0.26),
        "shank": rng.uniform(0.22, 0.26),
        "ankle_h": rng.uniform(0.03, 0.05),
    }


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def two_bone_ik(root: np.ndarray, end: np.ndarray, l1: float, l2: float,
                pole: np.ndarray) -> np.ndarray:
    """Middle joint of a two-bone chain bending towards `pole`.

    `root` and `end` are (T, 3); `pole` is (3,) or (T, 3). If `end` is out of
    reach it is pulled in along the root->end line, so returned bone lengths
    are exactly l1 and l2.
    """
    d_vec = end - root
    d = np.linalg.norm(d_vec, axis=-1, keepdims=True)
    d_clamped = np.clip(d, abs(l1 - l2) + 1e-4, l1 + l2 - 1e-5)
    u = d_vec / (d + 1e-9)
    pole = np.broadcast_to(pole, root.shape)
    p_perp = _unit(pole - np.sum(pole * u, axis=-1, keepdims=True) * u)
    cos_a = np.clip((l1 ** 2 + d_clamped ** 2 - l2 ** 2) / (2 * l1 * d_clamped), -1, 1)
    sin_a = np.sqrt(1 - cos_a ** 2)
    return root + l1 * (cos_a * u + sin_a * p_perp)


def _place_end(root: np.ndarray, end: np.ndarray, l1: float, l2: float,
               pole: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve IK and return (mid, reachable_end) with exact bone lengths."""
    mid = two_bone_ik(root, end, l1, l2, pole)
    end_fixed = mid + l2 * _unit(end - mid)
    return mid, end_fixed


def _arm_fk(shoulder: np.ndarray, pitch: np.ndarray, abd: np.ndarray, elbow_flex: np.ndarray,
            side: float, b: dict) -> tuple[np.ndarray, np.ndarray]:
    """Arm from shoulder angles. pitch: forward swing from hanging, abd: sideways raise."""
    def direction(p, a):
        return np.stack([side * np.sin(a), -np.cos(a) * np.cos(p), np.cos(a) * np.sin(p)], -1)
    elbow = shoulder + b["upper_arm"] * direction(pitch, abd)
    wrist = elbow + b["forearm"] * direction(pitch + elbow_flex, abd)
    return elbow, wrist


def _assemble(n_frames: int, **parts: np.ndarray) -> np.ndarray:
    out = np.zeros((n_frames, NUM_JOINTS, 3))
    for name, val in parts.items():
        out[:, J[name]] = val
    return out


# ----------------------------------------------------------------------- timing

def _phase_signal(rng: np.random.Generator, n: int, fps: float, period: float,
                  idle_head: int, idle_tail: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (phase in [0, 2pi), depth in [0, 1], active mask) for n frames.

    `phase` is true rep progress (the regression target). `depth` is the
    pose-space amount of movement, which lingers at the top and bottom of each
    rep (a held squat or a pause at lockout) via a monotone time warp.
    """
    active = np.zeros(n)
    active[idle_head:n - idle_tail] = 1.0
    # Tempo wanders smoothly around the nominal period.
    wobble = np.cumsum(rng.normal(0, 0.03, n))
    wobble = np.clip(wobble - wobble.mean(), -0.25, 0.25)
    rate = (1.0 + wobble) / (period * fps) * active
    progress = rng.uniform(0, 0.3) + np.cumsum(rate)  # in reps
    frac = progress % 1.0
    hold = rng.uniform(0.0, 0.85)
    warped = frac - hold * np.sin(4 * np.pi * frac) / (4 * np.pi)
    depth = (1 - np.cos(TWO_PI * warped)) / 2
    # Per-rep amplitude variation.
    rep_idx = np.floor(progress).astype(int)
    amps = rng.uniform(0.85, 1.05, rep_idx.max() + 2)
    depth = depth * amps[rep_idx]
    depth = depth * active
    return TWO_PI * frac, np.clip(depth, 0, 1.1), active


# -------------------------------------------------------------------- exercises

def _standing_arms(rng, n, sh_l, sh_r, b, t):
    pitch = rng.uniform(0.0, 0.25) + 0.05 * np.sin(t * rng.uniform(0.5, 1.5))
    flex = rng.uniform(0.1, 0.4)
    el_l, wr_l = _arm_fk(sh_l, np.full(n, pitch), np.full(n, 0.12), np.full(n, flex), 1.0, b)
    el_r, wr_r = _arm_fk(sh_r, np.full(n, pitch), np.full(n, 0.12), np.full(n, flex), -1.0, b)
    return el_l, wr_l, el_r, wr_r


def _torso_and_head(hip_mid, lean, b, side_tilt=None):
    up = np.stack([np.zeros_like(lean), np.cos(lean), np.sin(lean)], -1)
    if side_tilt is not None:
        up = _unit(up + np.stack([side_tilt, np.zeros_like(lean), np.zeros_like(lean)], -1))
    sh_mid = hip_mid + b["torso"] * up
    half_sh = np.array([b["shoulder_w"] / 2, 0, 0])
    nose = sh_mid + b["head"] * up + np.array([0, -0.01, 0.04])
    return sh_mid + half_sh, sh_mid - half_sh, nose  # +x is the person's left


def _squat(rng, n, depth, faults, b, t):
    leg = b["thigh"] + b["shank"]
    max_depth = rng.uniform(0.30, 0.45) if "partial_range" in faults else rng.uniform(0.85, 1.0)
    d = depth * max_depth
    stance = b["hip_w"] / 2 + rng.uniform(0.01, 0.06)
    ank_l = np.tile([stance, b["ankle_h"], 0.0], (n, 1))
    ank_r = np.tile([-stance, b["ankle_h"], 0.0], (n, 1))
    hip_y = b["ankle_h"] + leg * (0.985 - 0.55 * d)
    hip_z = -0.16 * d * leg
    hip_mid = np.stack([np.zeros(n), hip_y, hip_z], -1)
    hip_l = hip_mid + [b["hip_w"] / 2, 0, 0]
    hip_r = hip_mid - [b["hip_w"] / 2, 0, 0]
    pole = np.array([0.25, 0.0, 1.0])
    kn_l, ank_l = _place_end(hip_l, ank_l, b["thigh"], b["shank"], pole)
    kn_r, ank_r = _place_end(hip_r, ank_r, b["thigh"], b["shank"], pole * [-1, 1, 1])
    lean = rng.uniform(0.02, 0.12) + 0.55 * d + (0.55 * d + 0.15 * np.clip(4 * d, 0, 1) if "forward_lean" in faults else 0)
    sh_l, sh_r, nose = _torso_and_head(hip_mid, lean, b)
    arm_style = rng.integers(3)
    if arm_style == 0:  # arms raised forward for balance as you descend
        pitch, flex = 0.2 + 1.3 * d, np.full(n, 0.1)
    elif arm_style == 1:  # hands clasped at chest
        pitch, flex = np.full(n, 0.6), np.full(n, 2.2)
    else:  # arms forward throughout
        pitch, flex = np.full(n, 1.45), np.full(n, 0.05)
    el_l, wr_l = _arm_fk(sh_l, pitch, np.full(n, 0.1), flex, 1.0, b)
    el_r, wr_r = _arm_fk(sh_r, pitch, np.full(n, 0.1), flex, -1.0, b)
    return _assemble(n, nose=nose, left_shoulder=sh_l, right_shoulder=sh_r, left_elbow=el_l,
                     right_elbow=el_r, left_wrist=wr_l, right_wrist=wr_r, left_hip=hip_l,
                     right_hip=hip_r, left_knee=kn_l, right_knee=kn_r, left_ankle=ank_l,
                     right_ankle=ank_r)


def _lunge(rng, n, depth, faults, b, t):
    leg = b["thigh"] + b["shank"]
    max_depth = rng.uniform(0.25, 0.4) if "partial_range" in faults else rng.uniform(0.85, 1.0)
    d = depth * max_depth
    stride = rng.uniform(0.32, 0.42)
    front_left = rng.random() < 0.5
    sx = b["hip_w"] / 2
    front = np.tile([sx if front_left else -sx, b["ankle_h"], stride / 2], (n, 1))
    back = np.tile([-sx if front_left else sx, b["ankle_h"] + 0.04, -stride / 2], (n, 1))
    hip_y = b["ankle_h"] + leg * (0.93 - 0.42 * d)
    hip_mid = np.stack([np.zeros(n), hip_y, np.full(n, -0.02)], -1)
    hip_l = hip_mid + [sx, 0, 0]
    hip_r = hip_mid - [sx, 0, 0]
    pole = np.array([0.0, 0.0, 1.0])
    if front_left:
        kn_l, ank_l = _place_end(hip_l, front, b["thigh"], b["shank"], pole)
        kn_r, ank_r = _place_end(hip_r, back, b["thigh"], b["shank"], pole)
    else:
        kn_r, ank_r = _place_end(hip_r, front, b["thigh"], b["shank"], pole)
        kn_l, ank_l = _place_end(hip_l, back, b["thigh"], b["shank"], pole)
    lean = rng.uniform(0.0, 0.1) + 0.1 * d + (0.6 * d + 0.12 * np.clip(4 * d, 0, 1) if "forward_lean" in faults else 0)
    sh_l, sh_r, nose = _torso_and_head(hip_mid, lean, b)
    if rng.random() < 0.5:
        el_l, wr_l, el_r, wr_r = _standing_arms(rng, n, sh_l, sh_r, b, t)
    else:  # hands on hips
        pole = np.array([1.0, 0.0, -0.4])
        el_l, wr_l = _place_end(sh_l, hip_l + [0.03, 0.03, 0.0], b["upper_arm"], b["forearm"], pole)
        el_r, wr_r = _place_end(sh_r, hip_r + [-0.03, 0.03, 0.0], b["upper_arm"], b["forearm"],
                                pole * [-1, 1, 1])
    return _assemble(n, nose=nose, left_shoulder=sh_l, right_shoulder=sh_r, left_elbow=el_l,
                     right_elbow=el_r, left_wrist=wr_l, right_wrist=wr_r, left_hip=hip_l,
                     right_hip=hip_r, left_knee=kn_l, right_knee=kn_r, left_ankle=ank_l,
                     right_ankle=ank_r)


def _jumping_jack(rng, n, depth, faults, b, t):
    leg = b["thigh"] + b["shank"]
    s = np.clip(depth, 0, 1)
    arm_top = rng.uniform(1.2, 1.7) if "partial_range" in faults else rng.uniform(2.6, 3.0)
    abd = 0.12 + (arm_top - 0.12) * s
    foot_x = b["hip_w"] / 2 + 0.01 + rng.uniform(0.18, 0.26) * s
    hop = 0.035 * np.sin(np.pi * s) ** 2
    hip_mid = np.stack([np.zeros(n), b["ankle_h"] + leg * (0.985 - 0.06 * s) + hop, np.zeros(n)], -1)
    hip_l = hip_mid + [b["hip_w"] / 2, 0, 0]
    hip_r = hip_mid - [b["hip_w"] / 2, 0, 0]
    ank_l = np.stack([foot_x, b["ankle_h"] + hop, np.zeros(n)], -1)
    ank_r = np.stack([-foot_x, b["ankle_h"] + hop, np.zeros(n)], -1)
    pole = np.array([0.0, 0.0, 1.0])
    kn_l, ank_l = _place_end(hip_l, ank_l, b["thigh"], b["shank"], pole)
    kn_r, ank_r = _place_end(hip_r, ank_r, b["thigh"], b["shank"], pole)
    sh_l, sh_r, nose = _torso_and_head(hip_mid, np.full(n, rng.uniform(0.0, 0.08)), b)
    el_l, wr_l = _arm_fk(sh_l, np.full(n, 0.05), abd, np.full(n, 0.15), 1.0, b)
    el_r, wr_r = _arm_fk(sh_r, np.full(n, 0.05), abd, np.full(n, 0.15), -1.0, b)
    return _assemble(n, nose=nose, left_shoulder=sh_l, right_shoulder=sh_r, left_elbow=el_l,
                     right_elbow=el_r, left_wrist=wr_l, right_wrist=wr_r, left_hip=hip_l,
                     right_hip=hip_r, left_knee=kn_l, right_knee=kn_r, left_ankle=ank_l,
                     right_ankle=ank_r)


def _pushup(rng, n, depth, faults, b, t):
    """Side-plane push-up: toes at the origin, body pivoting about the feet."""
    leg = b["thigh"] + b["shank"]
    body = leg + b["torso"]
    arm = b["upper_arm"] + b["forearm"]
    top_h = arm * rng.uniform(0.93, 0.99)
    max_depth = rng.uniform(0.2, 0.35) if "partial_range" in faults else rng.uniform(0.8, 0.95)
    sh_h = top_h - (top_h - 0.09) * depth * max_depth  # shoulder height above floor
    beta = np.arcsin(np.clip((sh_h - b["ankle_h"]) / body, -1, 1))
    axis = np.stack([np.zeros(n), np.sin(beta), np.cos(beta)], -1)  # feet -> head
    normal = np.stack([np.zeros(n), np.cos(beta), -np.sin(beta)], -1)  # "up" off the back
    ank_mid = np.tile([0.0, b["ankle_h"], 0.0], (n, 1))
    bend = 0.0
    if "hip_sag" in faults:
        bend = -rng.uniform(0.05, 0.08)
    elif "hip_pike" in faults:
        bend = rng.uniform(0.07, 0.11)
    bend = bend * (0.6 + 0.4 * depth)[:, None]
    hip_mid = ank_mid + leg * axis + bend * normal
    sh_mid = ank_mid + body * axis
    # Re-normalise so the torso keeps its length after the hip moved.
    sh_mid = hip_mid + b["torso"] * _unit(sh_mid - hip_mid)
    knee_mid = ank_mid + b["shank"] * _unit(hip_mid - ank_mid)
    half_sh = np.array([b["shoulder_w"] / 2, 0, 0])
    half_hip = np.array([b["hip_w"] / 2, 0, 0])
    half_ft = np.array([b["hip_w"] / 2 - 0.02, 0, 0])
    nose = sh_mid + b["head"] * _unit(sh_mid - hip_mid) - 0.05 * normal
    hand_x = b["shoulder_w"] / 2 + rng.uniform(0.02, 0.08)
    hand_z = sh_mid[0, 2] + rng.uniform(-0.03, 0.03)
    hand_l = np.tile([hand_x, 0.02, hand_z], (n, 1))
    hand_r = np.tile([-hand_x, 0.02, hand_z], (n, 1))
    sh_l, sh_r = sh_mid + half_sh, sh_mid - half_sh
    pole_l = np.array([0.6, 0.2, -1.0])
    el_l, wr_l = _place_end(sh_l, hand_l, b["upper_arm"], b["forearm"], pole_l)
    el_r, wr_r = _place_end(sh_r, hand_r, b["upper_arm"], b["forearm"], pole_l * [-1, 1, 1])
    return _assemble(n, nose=nose, left_shoulder=sh_l, right_shoulder=sh_r, left_elbow=el_l,
                     right_elbow=el_r, left_wrist=wr_l, right_wrist=wr_r,
                     left_hip=hip_mid + half_hip, right_hip=hip_mid - half_hip,
                     left_knee=knee_mid + half_hip, right_knee=knee_mid - half_hip,
                     left_ankle=ank_mid + half_ft, right_ankle=ank_mid - half_ft)


def _idle(rng, n, b, t):
    """Standing around: sway, shifting weight, fidgeting arms, occasional step."""
    leg = b["thigh"] + b["shank"]
    f = rng.uniform(0.2, 0.8)
    sway = 0.02 * np.sin(TWO_PI * f * t + rng.uniform(0, TWO_PI))
    drift = np.cumsum(rng.normal(0, 0.002, n))
    hip_mid = np.stack([sway + drift, b["ankle_h"] + leg * rng.uniform(0.96, 0.99) + 0 * t,
                        0.01 * np.sin(TWO_PI * f * 0.7 * t)], -1)
    hip_l = hip_mid + [b["hip_w"] / 2, 0, 0]
    hip_r = hip_mid - [b["hip_w"] / 2, 0, 0]
    stance = b["hip_w"] / 2 + rng.uniform(0.0, 0.05)
    ank_l = np.stack([np.full(n, stance) + drift, np.full(n, b["ankle_h"]), np.zeros(n)], -1)
    ank_r = np.stack([np.full(n, -stance) + drift, np.full(n, b["ankle_h"]), np.zeros(n)], -1)
    pole = np.array([0.0, 0.0, 1.0])
    kn_l, ank_l = _place_end(hip_l, ank_l, b["thigh"], b["shank"], pole)
    kn_r, ank_r = _place_end(hip_r, ank_r, b["thigh"], b["shank"], pole)
    lean = rng.uniform(-0.05, 0.15) + 0.03 * np.sin(TWO_PI * f * 0.5 * t)
    sh_l, sh_r, nose = _torso_and_head(hip_mid, lean, b)
    # Random slow arm gestures (stretching, checking phone, hands on hips...).
    g = rng.uniform(0.1, 0.6)
    pitch_l = rng.uniform(0, 1.2) * (0.5 + 0.5 * np.sin(TWO_PI * g * t + rng.uniform(0, TWO_PI)))
    pitch_r = rng.uniform(0, 1.2) * (0.5 + 0.5 * np.sin(TWO_PI * g * t + rng.uniform(0, TWO_PI)))
    abd = np.full(n, rng.uniform(0.05, 0.5))
    el_l, wr_l = _arm_fk(sh_l, pitch_l, abd, np.full(n, rng.uniform(0.1, 1.8)), 1.0, b)
    el_r, wr_r = _arm_fk(sh_r, pitch_r, abd, np.full(n, rng.uniform(0.1, 1.8)), -1.0, b)
    return _assemble(n, nose=nose, left_shoulder=sh_l, right_shoulder=sh_r, left_elbow=el_l,
                     right_elbow=el_r, left_wrist=wr_l, right_wrist=wr_r, left_hip=hip_l,
                     right_hip=hip_r, left_knee=kn_l, right_knee=kn_r, left_ankle=ank_l,
                     right_ankle=ank_r)


_EXERCISE_FNS = {"squat": _squat, "lunge": _lunge, "jumping_jack": _jumping_jack, "pushup": _pushup}
_PERIOD = {"squat": (1.3, 3.2), "lunge": (1.5, 3.2), "jumping_jack": (0.6, 1.2), "pushup": (1.1, 2.8)}
# Camera yaw ranges (radians, 0 = person faces camera). Each exercise is filmed
# from angles a user would plausibly pick.
_YAW = {"squat": [(-1.9, 1.9)], "lunge": [(-1.9, -0.6), (0.6, 1.9), (-0.6, 0.6)],
        "jumping_jack": [(-0.6, 0.6)], "pushup": [(-2.2, -0.9), (0.9, 2.2)],
        "idle": [(-1.6, 1.6)]}


# ------------------------------------------------------------------- rendering

def render(rng: np.random.Generator, world: np.ndarray, exercise: str) -> np.ndarray:
    """Project world joints (T, 13, 3) to (T, 13, 3) image-space x, y, visibility."""
    n = world.shape[0]
    lo, hi = _YAW[exercise][rng.integers(len(_YAW[exercise]))]
    yaw = rng.uniform(lo, hi)
    elev = rng.uniform(-0.25, 0.3)
    cy, sy = np.cos(yaw), np.sin(yaw)
    ce, se = np.cos(elev), np.sin(elev)
    # Person rotates by yaw about +y, then the camera tilts by elevation about +x.
    rot_y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rot_x = np.array([[1, 0, 0], [0, ce, -se], [0, se, ce]])
    cam = world @ rot_y.T @ rot_x.T  # camera looks along -z; +z points at the lens
    aspect = rng.choice([4 / 3, 16 / 9, 3 / 4, 9 / 16, 1.0])
    span_y = np.ptp(cam[..., 1]) + 1e-6
    span_x = np.ptp(cam[..., 0]) + 1e-6
    scale = rng.uniform(0.45, 0.9) / max(span_y, span_x / aspect)
    # Weak perspective: slightly larger when closer to the camera.
    persp = 1.0 + rng.uniform(0.0, 0.25) * cam[..., 2]
    x = cam[..., 0] * scale * persp
    y = -cam[..., 1] * scale * persp
    x = x - (x.min() + x.max()) / 2 + aspect * rng.uniform(0.35, 0.65)
    y = y - (y.min() + y.max()) / 2 + rng.uniform(0.4, 0.6)
    if rng.random() < 0.5:  # selfie / mirrored camera
        x = aspect - x

    # --- pose-estimator corruption ---
    noise = rng.uniform(0.002, 0.012)
    x = x + rng.normal(0, noise, x.shape)
    y = y + rng.normal(0, noise, y.shape)
    drift = np.cumsum(rng.normal(0, noise * 0.15, (n, 1, 2)), axis=0)
    x, y = x + drift[..., 0], y + drift[..., 1]
    depth = cam[..., 2]
    z_rel = (depth - depth.mean(axis=1, keepdims=True)) / (np.ptp(depth) + 1e-6)
    vis = np.clip(rng.uniform(0.88, 0.99) + 0.25 * np.minimum(z_rel, 0) + rng.normal(0, 0.03, x.shape), 0, 1)
    if rng.random() < 0.35:  # transient occlusions
        for _ in range(rng.integers(1, 4)):
            j = rng.integers(NUM_JOINTS)
            a = rng.integers(0, n)
            bnd = min(n, a + rng.integers(3, max(4, n // 3)))
            vis[a:bnd, j] = rng.uniform(0.02, 0.4, bnd - a)
            x[a:bnd, j] += rng.normal(0, 0.03, bnd - a)
            y[a:bnd, j] += rng.normal(0, 0.03, bnd - a)
    out = np.stack([x, y, vis], -1)
    if rng.random() < 0.2:  # frame stalls: the estimator repeats a frame
        idx = np.arange(n)
        for _ in range(rng.integers(1, 4)):
            k = rng.integers(1, n)
            idx[k] = idx[k - 1]
        out = out[idx]
    return out.astype(np.float32)


# --------------------------------------------------------------------- sampling

def sample_clip(rng: np.random.Generator, n_frames: int = 64, exercise: str | None = None,
                faults: list[str] | None = None, fps: float | None = None,
                idle_margin: bool = True) -> Clip:
    if exercise is None:
        exercise = EXERCISES[rng.choice(len(EXERCISES), p=[0.2, 0.2, 0.2, 0.2, 0.2])]
    fps = fps if fps is not None else rng.uniform(14, 31)
    t = np.arange(n_frames) / fps
    body = sample_body(rng)
    fault_vec = np.zeros(len(FAULTS), dtype=np.float32)

    if exercise == "idle":
        world = _idle(rng, n_frames, body, t)
        phase = np.zeros(n_frames)
        active = np.zeros(n_frames)
    else:
        if faults is None:
            faults = [f for f in APPLICABLE_FAULTS[exercise] if rng.random() < 0.3]
            if "hip_sag" in faults and "hip_pike" in faults:
                faults.remove("hip_pike" if rng.random() < 0.5 else "hip_sag")
        for f in faults:
            fault_vec[FAULTS.index(f)] = 1.0
        head = tail = 0
        if idle_margin and rng.random() < 0.3:
            head = rng.integers(0, n_frames // 4)
        if idle_margin and rng.random() < 0.15:
            tail = rng.integers(0, n_frames // 4)
        period = rng.uniform(*_PERIOD[exercise])
        phase, depth, active = _phase_signal(rng, n_frames, fps, period, head, tail)
        world = _EXERCISE_FNS[exercise](rng, n_frames, depth, faults, body, t)
    lm = render(rng, world, exercise)
    return Clip(lm, EXERCISES.index(exercise), fault_vec, phase.astype(np.float32),
                active.astype(np.float32))


def sample_batch(rng: np.random.Generator, batch: int, n_frames: int = 64) -> dict[str, np.ndarray]:
    clips = [sample_clip(rng, n_frames) for _ in range(batch)]
    return {
        "landmarks": np.stack([c.landmarks for c in clips]),
        "exercise": np.array([c.exercise for c in clips], dtype=np.int64),
        "faults": np.stack([c.faults for c in clips]),
        "phase": np.stack([c.phase for c in clips]),
        "active": np.stack([c.active for c in clips]),
    }
