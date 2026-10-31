"""Regression tests for the audit correctness fixes."""

from __future__ import annotations

import numpy as np

from src.actions import annotate_actions
from src.factory import factory_metrics
from src.scorecard import build_scorecard
from tests.test_scale import canonical_hand, make_hand
from tests.test_trust import _blank, _chain


def _action_hand(curl: float, *, inferred: bool = False, wrist_z: float = 0.0) -> dict:
    points = _blank()
    _chain(points, curl)
    points[0] = [0.0, 0.0, wrist_z]
    return {
        "label": "Right",
        "world": points.tolist(),
        "world_fixed": points.tolist(),
        "inferred": inferred,
        "joint_confidence": [0.9] * 21,
        "score": 0.9,
    }


def test_procrustes_alignment_removes_pose_and_scale():
    world = canonical_hand()
    angle = 0.6
    rot = np.array([
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ])
    other = 1.8 * ((rot @ world.T).T) + np.array([0.4, -0.2, 0.7])
    frames = [{"hands": [{"label": "Left", "world": world.tolist(), "mesh": {"world": other.tolist()}}]}]
    agreement = build_scorecard(frames, {})["mediapipe_wilor"]
    assert agreement["frames"] == 1
    assert agreement["mean_joint_m"] < 1e-6
    assert agreement["mean_joint_raw_m"] > 0.1
    assert abs(agreement["mean_scale"] - 1.0 / 1.8) < 1e-3


def test_no_grasp_event_when_hand_reappears_after_gap():
    frames = []
    for index in range(3):
        frames.append({"frame": index, "t_sec": index / 30.0, "hands": [_action_hand(0.0)]})
    for index in range(3, 10):
        frames.append({"frame": index, "t_sec": index / 30.0, "hands": []})
    frames.append({"frame": 10, "t_sec": 10 / 30.0, "hands": [_action_hand(0.08)]})
    annotate_actions(frames, fps=30.0)
    assert frames[10]["hands"][0]["action"] != "grasp"
    assert frames[10]["hands"][0]["action"] == "idle"


def test_no_grasp_event_on_inferred_frames():
    frames = [
        {"frame": 0, "t_sec": 0.0, "hands": [_action_hand(0.0)]},
        {"frame": 1, "t_sec": 1 / 30.0, "hands": [_action_hand(0.06, inferred=True)]},
    ]
    annotate_actions(frames, fps=30.0)
    assert frames[1]["hands"][0]["action"] != "grasp"


def _controlled_curl_hand(target_curl: float) -> dict:
    """Hand whose five finger chains all measure exactly target_curl radians."""
    points = np.zeros((21, 3), dtype=np.float64)
    for chain_index, (a, b, c) in enumerate(((5, 6, 8), (9, 10, 12), (13, 14, 16), (17, 18, 20), (1, 2, 4))):
        x = 0.01 * chain_index
        theta = np.pi - target_curl
        points[b] = [x, 0.04, 0.0]
        points[a] = [x, 0.02, 0.0]
        points[c] = [x + 0.02 * np.sin(theta), 0.04 - 0.02 * np.cos(theta), 0.0]
    return {
        "label": "Right",
        "world": points.tolist(),
        "world_fixed": points.tolist(),
        "joint_confidence": [0.9] * 21,
        "score": 0.9,
    }


def test_irregular_timestamps_use_real_delta():
    slow_frames = [
        {"frame": 0, "t_sec": 0.0, "hands": [_controlled_curl_hand(0.5)]},
        {"frame": 1, "t_sec": 1.0, "hands": [_controlled_curl_hand(1.3)]},
    ]
    annotate_actions(slow_frames, fps=30.0)
    # 0.8 rad of curl over a full second is not a grasp event.
    assert slow_frames[1]["hands"][0]["action"] != "grasp"
    fast_frames = [
        {"frame": 0, "t_sec": 0.0, "hands": [_controlled_curl_hand(0.5)]},
        {"frame": 1, "t_sec": 1 / 30.0, "hands": [_controlled_curl_hand(1.3)]},
    ]
    annotate_actions(fast_frames, fps=30.0)
    # The same change inside one 30 fps frame interval is a grasp.
    assert fast_frames[1]["hands"][0]["action"] == "grasp"


def test_factory_excludes_low_trust_and_inferred_from_shares():
    frames = []
    for index in range(4):
        hand = _action_hand(0.0)
        hand["joint_confidence"] = [0.1] * 21
        hand["world_fixed"] = canonical_hand().tolist()
        hand["world_fixed"][9] = [0.0, 0.3, 0.01]
        frames.append({"frame": index, "t_sec": index / 30.0, "hands": [hand]})
    report = factory_metrics(frames, [], fps=30.0)
    assert report["trusted_filter"]["excluded_hands"] == 4
    assert report["risky_wrist_share"]["Right"] is None
    trusted_hand = _action_hand(0.0)
    trusted_hand["world_fixed"] = canonical_hand().tolist()
    trusted_hand["world_fixed"][9] = [0.0, 0.3, 0.01]
    frames.append({"frame": 4, "t_sec": 4 / 30.0, "hands": [trusted_hand]})
    report = factory_metrics(frames, [], fps=30.0)
    assert report["trusted_filter"]["excluded_hands"] == 4
    assert report["risky_wrist_share"]["Right"] is not None
    inferred_hand = _action_hand(0.0, inferred=True)
    frames.append({"frame": 5, "t_sec": 5 / 30.0, "hands": [inferred_hand]})
    report = factory_metrics(frames, [], fps=30.0)
    assert report["trusted_filter"]["excluded_hands"] == 5
