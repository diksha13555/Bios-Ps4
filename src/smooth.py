"""Short gap fill between real detections."""

from __future__ import annotations

import numpy as np


def _lerp(a, b, t: float):
    return ((1.0 - t) * np.asarray(a, dtype=np.float64) + t * np.asarray(b, dtype=np.float64)).tolist()


def _lerp_hand(start: dict, end: dict, t: float) -> dict:
    return {
        "label": start["label"],
        "score": float((1.0 - t) * start["score"] + t * end["score"]),
        "world": _lerp(start["world"], end["world"], t),
        "image": _lerp(start["image"], end["image"], t),
        "tracked": False,
        "interpolated": True,
    }


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
            for step, index in enumerate(range(left + 1, right), start=1):
                if any(h["label"] == label for h in frames[index]["hands"]):
                    continue
                t = step / float(gap + 1)
                frames[index]["hands"].append(_lerp_hand(start, end, t))
    for frame in frames:
        order = {"Left": 0, "Right": 1}
        frame["hands"].sort(key=lambda hand: order.get(hand["label"], 9))
    return frames
