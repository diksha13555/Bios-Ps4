"""Cycle time, hand balance, a RULA-style wrist score, and motion smoothness.

The wrist score uses the metacarpal direction in the fixed world frame and the
RULA wrist bands (neutral, within 15 degrees, beyond 15 degrees). It is the
wrist row of RULA, not a whole-body assessment.
"""

from __future__ import annotations

import numpy as np


def _wrist_angles(hand: dict) -> tuple[float, float]:
    points = np.asarray(hand.get("world_fixed") or hand["world"], dtype=np.float64)
    metacarpal = points[9] - points[0]
    horizontal = float(np.hypot(metacarpal[0], metacarpal[2]))
    flexion = float(np.degrees(np.arctan2(metacarpal[1], max(horizontal, 1e-8))))
    across = points[5] - points[17]
    deviation = float(np.degrees(np.arctan2(across[1], max(float(np.hypot(across[0], across[2])), 1e-8))))
    return abs(flexion), abs(deviation)


def _rula_wrist(flexion: float, deviation: float) -> int:
    if flexion <= 1.0:
        score = 1
    elif flexion <= 15.0:
        score = 2
    else:
        score = 3
    if deviation > 15.0:
        score += 1
    return score


def _smoothness(samples: list[tuple[float, np.ndarray]]) -> float | None:
    if len(samples) < 4:
        return None
    times = np.asarray([item[0] for item in samples], dtype=np.float64)
    positions = np.stack([item[1] for item in samples])
    dt = np.diff(times)
    dt = np.maximum(dt, 1e-4)
    velocity = np.diff(positions, axis=0) / dt[:, None]
    if len(velocity) < 3:
        return None
    mid_dt = np.maximum(0.5 * (dt[:-1] + dt[1:]), 1e-4)
    acceleration = np.diff(velocity, axis=0) / mid_dt[:, None]
    jerk_dt = np.maximum(0.5 * (mid_dt[:-1] + mid_dt[1:]), 1e-4) if len(mid_dt) > 1 else mid_dt
    if len(acceleration) < 2:
        return None
    jerk = np.diff(acceleration, axis=0) / jerk_dt[: len(acceleration) - 1, None]
    duration = float(times[-1] - times[0])
    path = float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum())
    if duration < 1e-3 or path < 1e-6:
        return None
    normalized = float(np.sqrt(np.mean(np.sum(jerk * jerk, axis=1)))) * duration**3 / path
    return float(1.0 / (1.0 + normalized))


def _trusted(hand: dict, min_trust: float) -> bool:
    """Exclude inferred/low-confidence frames from factory statistics."""
    if hand.get("inferred"):
        return False
    confidence = hand.get("joint_confidence") or []
    if not confidence:
        return True
    return float(np.mean(confidence)) >= min_trust


def factory_metrics(frames: list[dict], segments: list[dict], fps: float, *, min_trust: float = 0.35) -> dict:
    active = {"Left": 0, "Right": 0}
    risky = {"Left": 0, "Right": 0}
    scored = {"Left": 0, "Right": 0}
    paths: dict[str, list[tuple[float, np.ndarray]]] = {"Left": [], "Right": []}
    total_hands = 0
    excluded_hands = 0
    for frame in frames:
        for hand in frame["hands"]:
            label = hand["label"]
            if label not in active:
                continue
            total_hands += 1
            flexion, deviation = _wrist_angles(hand)
            hand["rula_wrist"] = _rula_wrist(flexion, deviation)
            if not _trusted(hand, min_trust):
                excluded_hands += 1
                continue
            if hand.get("action") != "idle":
                active[label] += 1
            scored[label] += 1
            if hand["rula_wrist"] >= 3:
                risky[label] += 1
            points = np.asarray(hand.get("world_fixed") or hand["world"], dtype=np.float64)
            paths[label].append((float(frame["t_sec"]), points[0, :3]))
    cycles = []
    for label in ("Left", "Right"):
        order = [item for item in segments if item["label"] == label]
        for index in range(len(order) - 3):
            names = [item["action"] for item in order[index : index + 4]]
            if names == ["reach", "grasp", "hold", "release"]:
                duration = float(order[index + 3]["end_sec"] - order[index]["start_sec"])
                cycles.append({"label": label, "start_sec": order[index]["start_sec"], "duration_sec": round(duration, 3)})
    total_active = active["Left"] + active["Right"]
    return {
        "cycle_count": len(cycles),
        "mean_cycle_sec": None if not cycles else round(float(np.mean([item["duration_sec"] for item in cycles])), 3),
        "cycles": cycles,
        "hand_balance": {
            "left_active_share": None if total_active == 0 else round(active["Left"] / total_active, 4),
            "right_active_share": None if total_active == 0 else round(active["Right"] / total_active, 4),
        },
        "risky_wrist_share": {
            label: None if scored[label] == 0 else round(risky[label] / scored[label], 4) for label in ("Left", "Right")
        },
        "smoothness": {
            label: None if (value := _smoothness(paths[label])) is None else round(value, 4)
            for label in ("Left", "Right")
        },
        "fps": fps,
        "trusted_filter": {
            "min_mean_joint_confidence": min_trust,
            "excluded_hands": excluded_hands,
            "excluded_share": None if total_hands == 0 else round(excluded_hands / total_hands, 4),
            "note": "Active share, RULA wrist share, and smoothness use trusted (non-inferred) frames only.",
        },
        "rula": "Wrist row only: 1 neutral, 2 within 15 deg, 3 beyond 15 deg, +1 if deviation exceeds 15 deg.",
    }
