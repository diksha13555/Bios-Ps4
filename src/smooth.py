"""Short gap fill between real detections."""

from __future__ import annotations

import numpy as np


def _lerp(a, b, t: float):
    return ((1.0 - t) * np.asarray(a, dtype=np.float64) + t * np.asarray(b, dtype=np.float64)).tolist()


def _lerp_hand(start: dict, end: dict, t: float, *, world=None, image=None, trust_cap: float = 0.35) -> dict:
    hand = {
        "label": start["label"],
        "score": min(trust_cap, float((1.0 - t) * start["score"] + t * end["score"])),
        "world": _lerp(start["world"], end["world"], t) if world is None else world,
        "image": _lerp(start["image"], end["image"], t) if image is None else image,
        "tracked": False,
        "interpolated": True,
        "inferred": True,
    }
    return hand


def _bridge(start: dict, end: dict, prev: dict | None, t: float, step: int) -> dict:
    """Short bridge. Motion history is used only when it agrees with the linear fill."""
    linear_w = np.asarray(_lerp(start["world"], end["world"], t), dtype=np.float64)
    linear_i = np.asarray(_lerp(start["image"], end["image"], t), dtype=np.float64)
    trust_cap = 0.35
    world = linear_w
    if prev is not None:
        velocity = np.asarray(start["world"], dtype=np.float64) - np.asarray(prev["world"], dtype=np.float64)
        extrapolated = np.asarray(start["world"], dtype=np.float64) + velocity * step
        span = float(np.linalg.norm(linear_w - np.asarray(start["world"], dtype=np.float64))) + 1e-6
        if float(np.linalg.norm(extrapolated - linear_w)) / span <= 0.5:
            world = 0.5 * linear_w + 0.5 * extrapolated
        else:
            trust_cap = 0.2
    return _lerp_hand(start, end, t, world=world.tolist(), image=linear_i.tolist(), trust_cap=trust_cap)


def interpolate_gaps(frames: list[dict], max_gap: int = 5) -> list[dict]:
    labels = {hand["label"] for frame in frames for hand in frame["hands"]}
    series: dict[str, dict[int, dict]] = {label: {} for label in labels}
    for index, frame in enumerate(frames):
        for hand in frame["hands"]:
            hand.setdefault("tracked", not hand.get("interpolated", False))
            hand.setdefault("interpolated", False)
            series[hand["label"]][index] = hand
    for label, points in series.items():
        indexes = sorted(points)
        for left, right in zip(indexes, indexes[1:]):
            gap = right - left - 1
            if gap < 1 or gap > max_gap:
                continue
            start, end = points[left], points[right]
            prev = points.get(left - 1)
            for step, index in enumerate(range(left + 1, right), start=1):
                if any(h["label"] == label for h in frames[index]["hands"]):
                    continue
                t = step / float(gap + 1)
                frames[index]["hands"].append(_bridge(start, end, prev, t, step))
    for frame in frames:
        order = {"Left": 0, "Right": 1}
        frame["hands"].sort(key=lambda hand: order.get(hand["label"], 9))
    return frames


def smooth_by_confidence(frames: list[dict], window: int = 5) -> None:
    """Pull low-trust joints toward a short temporal average. High-trust joints stay put."""
    labels = {hand["label"] for frame in frames for hand in frame["hands"]}
    half = max(window // 2, 1)
    for label in labels:
        located = []
        for frame in frames:
            for hand in frame["hands"]:
                if hand["label"] == label and hand.get("joint_confidence"):
                    located.append(hand)
        if len(located) < 2:
            continue
        worlds = np.stack([np.asarray(hand["world"], dtype=np.float64) for hand in located])
        trust = np.clip(np.stack([np.asarray(hand["joint_confidence"], dtype=np.float64) for hand in located]), 0.0, 1.0)
        averaged = np.empty_like(worlds)
        for index in range(len(located)):
            averaged[index] = worlds[max(0, index - half) : index + half + 1].mean(axis=0)
        blended = trust[..., None] * worlds + (1.0 - trust)[..., None] * averaged
        for hand, points in zip(located, blended):
            hand["world"] = points.tolist()
