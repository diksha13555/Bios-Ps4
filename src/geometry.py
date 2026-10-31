"""Bone lengths and bend limits.

Lengths are learned from trusted frames, then low-trust joints are pulled back
onto that hand. Fingers are not allowed to curl past a physiological stop.
"""

from __future__ import annotations

import numpy as np

from src.export_sim import FINGER_CHAINS

PARENT = {
    1: 0, 2: 1, 3: 2, 4: 3,
    5: 0, 6: 5, 7: 6, 8: 7,
    9: 0, 10: 9, 11: 10, 12: 11,
    13: 0, 14: 13, 15: 14, 16: 15,
    17: 0, 18: 17, 19: 18, 20: 19,
}
MAX_CURL = {"index": 2.4, "middle": 2.4, "ring": 2.4, "pinky": 2.4, "thumb": 2.0}
TRUST_FOR_TEMPLATE = 0.7


def _lengths(points: np.ndarray) -> dict[int, float]:
    lengths = {}
    for child, parent in PARENT.items():
        lengths[child] = float(np.linalg.norm(points[child] - points[parent]))
    return lengths


def _template(frames: list[dict], label: str) -> dict[int, float] | None:
    samples: dict[int, list[float]] = {child: [] for child in PARENT}
    for frame in frames:
        for hand in frame["hands"]:
            if hand["label"] != label or hand.get("inferred"):
                continue
            trust = hand.get("joint_confidence") or []
            if not trust:
                continue
            points = np.asarray(hand["world"], dtype=np.float64)
            for child, length in _lengths(points).items():
                if child >= len(trust) or float(trust[child]) < TRUST_FOR_TEMPLATE:
                    continue
                if length > 1e-6:
                    samples[child].append(length)
    template = {}
    for child, values in samples.items():
        if values:
            template[child] = float(np.median(values))
    return template or None


def _limit_curl(points: np.ndarray) -> None:
    names = ("index", "middle", "ring", "pinky", "thumb")
    for name, (proximal, joint, tip) in zip(names, FINGER_CHAINS):
        bone = points[tip] - points[joint]
        length = float(np.linalg.norm(bone))
        incoming = points[joint] - points[proximal]
        incoming_len = float(np.linalg.norm(incoming))
        if length < 1e-6 or incoming_len < 1e-6:
            continue
        incoming = incoming / incoming_len
        bone_dir = bone / length
        cos = float(np.clip(np.dot(-incoming, bone_dir), -1.0, 1.0))
        curl = float(np.pi - np.arccos(cos))
        limit = MAX_CURL[name]
        if curl <= limit:
            continue
        # Place the tip so the joint angle equals the limit, keeping bone length.
        axis = np.cross(incoming, bone_dir)
        if float(np.linalg.norm(axis)) < 1e-6:
            axis = np.array([0.0, 0.0, 1.0])
        axis = axis / np.linalg.norm(axis)
        # curl = pi - angle(incoming reversed, bone). Target bone direction from Rodrigues.
        target_angle = np.pi - limit
        k = axis
        v = -incoming
        rotated = (
            v * np.cos(target_angle)
            + np.cross(k, v) * np.sin(target_angle)
            + k * np.dot(k, v) * (1.0 - np.cos(target_angle))
        )
        points[tip] = points[joint] + rotated * length


def _project(points: np.ndarray, template: dict[int, float], trust: list[float]) -> np.ndarray:
    corrected = points.copy()
    for child, parent in PARENT.items():
        target = template.get(child)
        if target is None:
            continue
        joint_trust = float(trust[child]) if child < len(trust) else 1.0
        direction = corrected[child] - corrected[parent]
        length = float(np.linalg.norm(direction))
        if length < 1e-6:
            continue
        error = abs(length - target) / max(target, 1e-6)
        if joint_trust >= 0.55 and error < 0.25:
            continue
        corrected[child] = corrected[parent] + direction / length * target
    _limit_curl(corrected)
    return corrected


def enforce_hand_shape(frames: list[dict]) -> dict[str, dict[int, float]]:
    templates = {}
    labels = {hand["label"] for frame in frames for hand in frame["hands"]}
    for label in labels:
        template = _template(frames, label)
        if template is None:
            continue
        templates[label] = template
        for frame in frames:
            for hand in frame["hands"]:
                if hand["label"] != label:
                    continue
                points = np.asarray(hand["world"], dtype=np.float64)
                if points.shape[0] < 21:
                    continue
                hand["world"] = _project(points, template, hand.get("joint_confidence") or []).tolist()
    return templates
