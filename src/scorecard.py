"""Self-check without labelled ground truth."""

from __future__ import annotations

import numpy as np

from src.geometry import PARENT


def _bone_cv(frames: list[dict]) -> float | None:
    values: dict[tuple[str, int], list[float]] = {}
    for frame in frames:
        for hand in frame["hands"]:
            if hand.get("inferred"):
                continue
            points = np.asarray(hand["world"], dtype=np.float64)
            if points.shape[0] < 21:
                continue
            for child, parent in PARENT.items():
                length = float(np.linalg.norm(points[child] - points[parent]))
                if length > 1e-6:
                    values.setdefault((hand["label"], child), []).append(length)
    ratios = []
    for lengths in values.values():
        if len(lengths) < 3:
            continue
        mean = float(np.mean(lengths))
        if mean > 1e-6:
            ratios.append(float(np.std(lengths) / mean))
    if not ratios:
        return None
    return round(float(np.mean(ratios)), 4)


def _swaps(frames: list[dict]) -> int:
    count = 0
    for frame in frames:
        for hand in frame["hands"]:
            raw = hand.get("raw_label")
            if raw and raw != hand["label"] and not hand.get("inferred"):
                count += 1
    return count


def _wilor_agreement(frames: list[dict]) -> dict:
    distances = []
    for frame in frames:
        for hand in frame["hands"]:
            mesh = hand.get("mesh")
            if not isinstance(mesh, dict) or not mesh.get("world"):
                continue
            world = np.asarray(hand["world"], dtype=np.float64)
            other = np.asarray(mesh["world"], dtype=np.float64)
            count = min(len(world), len(other), 21)
            if count < 21:
                continue
            distances.append(float(np.linalg.norm(world[:count] - other[:count], axis=1).mean()))
    if not distances:
        return {"frames": 0, "mean_joint_m": None}
    return {"frames": len(distances), "mean_joint_m": round(float(np.mean(distances)), 5)}


def build_scorecard(frames: list[dict], drift: dict) -> dict:
    return {
        "bone_length_cv": _bone_cv(frames),
        "identity_disagreements": _swaps(frames),
        "mediapipe_wilor": _wilor_agreement(frames),
        "world_drift": drift,
        "notes": {
            "bone_length_cv": "Mean coefficient of variation of kinematic bone lengths. Lower is more stable.",
            "identity_disagreements": "Frames where MediaPipe's raw Left/Right disagreed with the locked identity.",
            "mediapipe_wilor": "Mean joint distance when a WiLoR mesh is attached. Null until a mesh merge exists.",
            "world_drift": "Wrist gap between a release and the next grasp. Translation is up to scale.",
        },
    }
