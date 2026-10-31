"""One 0–1 trust value per joint.

MediaPipe's hand score, per-joint reprojection error, temporal jitter, and
disagreement with a WiLoR mesh are combined when each cue is present.
Inferred frames are capped so a fill is never reported as a measurement.
"""

from __future__ import annotations

import math

import numpy as np

INFERRED_CAP = 0.35


def _unit(value: float, scale: float) -> float:
    return float(math.exp(-max(value, 0.0) / scale))


def _mesh_world(hand: dict) -> np.ndarray | None:
    mesh = hand.get("mesh")
    if not isinstance(mesh, dict) or not mesh.get("world"):
        return None
    points = np.asarray(mesh["world"], dtype=np.float64)
    if points.shape[0] < 21 or points.shape[1] < 3:
        return None
    return points[:21, :3]


def attach_confidence(frames: list[dict], fps: float) -> None:
    dt = 1.0 / max(float(fps), 1e-6)
    previous: dict[str, np.ndarray] = {}
    for frame in frames:
        for hand in frame["hands"]:
            world = np.asarray(hand["world"], dtype=np.float64)
            if world.ndim != 2 or world.shape[0] < 21:
                hand["joint_confidence"] = [0.0] * 21
                continue
            count = min(21, world.shape[0])
            score = float(np.clip(hand.get("score", 0.0), 0.0, 1.0))
            reproj = hand.get("joint_reproj_px") or []
            mesh = _mesh_world(hand)
            prior = previous.get(hand["label"])
            trusts = []
            for joint in range(count):
                terms = [score]
                weights = [0.35]
                if joint < len(reproj) and reproj[joint] is not None:
                    terms.append(_unit(float(reproj[joint]), 12.0))
                    weights.append(0.25)
                if prior is not None:
                    speed = float(np.linalg.norm(world[joint, :3] - prior[joint, :3]) / dt)
                    terms.append(_unit(speed, 0.15))
                    weights.append(0.20)
                if mesh is not None:
                    gap = float(np.linalg.norm(world[joint, :3] - mesh[joint]))
                    terms.append(_unit(gap, 0.02))
                    weights.append(0.20)
                total = sum(weights)
                trusts.append(float(np.clip(sum(w * t for w, t in zip(weights, terms)) / total, 0.0, 1.0)))
            if hand.get("inferred") or hand.get("interpolated"):
                trusts = [min(value, INFERRED_CAP) for value in trusts]
            hand["joint_confidence"] = [round(value, 4) for value in trusts] + [0.0] * (21 - count)
            previous[hand["label"]] = world[:count, :3].copy()
