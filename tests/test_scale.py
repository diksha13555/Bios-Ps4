"""Ego-scale recovery from hand anthropometrics."""

from __future__ import annotations

import numpy as np

from src.geometry import PARENT
from src.scale import REFERENCE_BONE_M, estimate_hand_scale, estimate_vo_scale


def canonical_hand(scale: float = 1.0) -> np.ndarray:
    lengths = {child: 0.02 for child in PARENT}
    lengths.update(REFERENCE_BONE_M)
    points = {0: np.zeros(3)}
    for child in sorted(PARENT):
        parent = PARENT[child]
        direction = np.array([1.0, 0.1 * (child % 3), 0.05 * (child % 2)])
        direction = direction / np.linalg.norm(direction)
        points[child] = points[parent] + direction * lengths[child] * scale
    return np.stack([points[i] for i in range(21)])


def make_hand(points, label="Left", trust=1.0, action=None, camera=None, inferred=False):
    hand = {
        "label": label,
        "score": trust,
        "tracked": True,
        "interpolated": False,
        "inferred": inferred,
        "raw_label": label,
        "world": np.asarray(points).tolist(),
        "image": np.asarray(points).tolist(),
        "joint_confidence": [trust] * 21,
        "mean_curl": 0.5,
    }
    if action is not None:
        hand["action"] = action
    if camera is not None:
        hand["camera"] = np.asarray(camera).tolist()
    return hand


def make_frame(index, hands, pose=None):
    frame = {"frame": index, "t_sec": round(index / 30.0, 4), "hands": hands}
    if pose is not None:
        frame["camera_pose_world"] = pose
    return frame


def pose(translation, valid=True, inliers=40):
    return {
        "rotation": np.eye(3).tolist(),
        "translation": list(translation),
        "valid": valid,
        "inliers": inliers,
    }


def test_hand_scale_recovers_oversized_hand():
    frames = [make_frame(i, [make_hand(canonical_hand(scale=1.25))]) for i in range(6)]
    result = estimate_hand_scale(frames)
    assert result["scale"] is not None
    assert abs(result["scale"] - 0.8) < 0.02
    assert result["per_label"]["Left"] == result["scale"]
    assert result["bones_used"] >= 4


def test_hand_scale_metric_hand_is_near_one():
    frames = [make_frame(i, [make_hand(canonical_hand(scale=1.0))]) for i in range(6)]
    result = estimate_hand_scale(frames)
    assert result["scale"] is not None
    assert abs(result["scale"] - 1.0) < 0.01


def test_hand_scale_none_when_insufficient_samples():
    frames = [make_frame(i, [make_hand(canonical_hand())]) for i in range(2)]
    assert estimate_hand_scale(frames)["scale"] is None


def test_hand_scale_skips_low_trust_and_inferred():
    frames = [make_frame(i, [make_hand(canonical_hand(scale=1.25), trust=0.4)]) for i in range(6)]
    assert estimate_hand_scale(frames)["scale"] is None
    frames = [make_frame(i, [make_hand(canonical_hand(scale=1.25), inferred=True)]) for i in range(6)]
    assert estimate_hand_scale(frames)["scale"] is None


def test_vo_scale_from_stationary_contacts():
    frames = []
    for i in range(10):
        camera = canonical_hand()
        camera[:, 0] -= 0.01 * i
        hand = make_hand(camera, action="hold", camera=camera)
        frames.append(make_frame(i, [hand], pose([0.005 * i, 0.0, 0.0])))
    result = estimate_vo_scale(frames)
    assert result["vo_scale"] is not None
    assert abs(result["vo_scale"] - 2.0) < 0.05
    assert result["pairs"] >= 5


def test_vo_scale_none_without_contacts():
    frames = []
    for i in range(10):
        camera = canonical_hand()
        camera[:, 0] -= 0.01 * i
        hand = make_hand(camera, action="reach", camera=camera)
        frames.append(make_frame(i, [hand], pose([0.005 * i, 0.0, 0.0])))
    assert estimate_vo_scale(frames)["vo_scale"] is None
