import numpy as np

from src.actions import annotate_actions
from src.confidence import attach_confidence
from src.factory import _rula_wrist
from src.geometry import enforce_hand_shape
from src.hands import OneEuroFilter
from src.smooth import interpolate_gaps, smooth_by_confidence
from src.world import putdown_pickup_drift


def _blank():
    return np.zeros((21, 3), dtype=np.float64)


def _chain(points, curl):
    points[5] = [0.0, 0.02, 0.0]
    points[6] = [0.0, 0.04, 0.0]
    points[8] = [0.0, 0.04 - curl, 0.0]
    points[9] = [0.01, 0.02, 0.0]
    points[10] = [0.01, 0.04, 0.0]
    points[12] = [0.01, 0.04 - curl, 0.0]
    points[13] = [0.02, 0.02, 0.0]
    points[14] = [0.02, 0.04, 0.0]
    points[16] = [0.02, 0.04 - curl, 0.0]
    points[17] = [0.03, 0.02, 0.0]
    points[18] = [0.03, 0.04, 0.0]
    points[20] = [0.03, 0.04 - curl, 0.0]
    points[1] = [0.0, 0.01, 0.01]
    points[2] = [0.0, 0.02, 0.02]
    points[4] = [0.0, 0.02 - curl, 0.03]


def test_low_trust_is_smoothed_more():
    trusted = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    shaky = OneEuroFilter(min_cutoff=1.0, beta=0.0)
    trusted(np.zeros(3), 0.0)
    shaky(np.zeros(3), 0.0)
    high = trusted(np.ones(3), 1.0, trust=1.0)
    low = shaky(np.ones(3), 1.0, trust=0.05)
    assert np.linalg.norm(low) < np.linalg.norm(high)


def test_reprojection_lowers_joint_trust():
    points = _blank()
    points[:, 0] = np.linspace(0, 0.1, 21)
    hand = {"label": "Left", "score": 0.9, "world": points.tolist(), "joint_reproj_px": [0.0] * 20 + [40.0]}
    attach_confidence([{"hands": [hand]}], fps=30.0)
    assert hand["joint_confidence"][0] > hand["joint_confidence"][20]


def test_inferred_gap_is_capped_and_long_gap_is_left_empty():
    start = {"label": "Left", "score": 0.9, "world": _blank().tolist(), "image": _blank().tolist(), "tracked": True, "interpolated": False}
    end = {"label": "Left", "score": 0.9, "world": (_blank() + 0.01).tolist(), "image": _blank().tolist(), "tracked": True, "interpolated": False}
    short = [{"frame": 0, "t_sec": 0.0, "hands": [dict(start)]}, {"frame": 1, "t_sec": 0.03, "hands": []}, {"frame": 2, "t_sec": 0.06, "hands": [dict(end)]}]
    interpolate_gaps(short, max_gap=5)
    assert short[1]["hands"][0]["inferred"] is True
    assert short[1]["hands"][0]["score"] <= 0.35
    long = [{"frame": i, "t_sec": i / 30, "hands": [dict(start)] if i in {0, 8} else []} for i in range(9)]
    interpolate_gaps(long, max_gap=5)
    assert long[4]["hands"] == []


def test_bone_lengths_are_pulled_back():
    good = _blank()
    for child, parent in ((1, 0), (2, 1)):
        good[child] = good[parent] + np.array([0.02, 0.0, 0.0])
    good[3] = good[2] + np.array([0.02, 0.0, 0.0])
    good[4] = good[3] + np.array([0.02, 0.0, 0.0])
    for root in (5, 9, 13, 17):
        good[root] = [0.02, 0.0, 0.0]
        good[root + 1] = [0.04, 0.0, 0.0]
        good[root + 2] = [0.06, 0.0, 0.0]
        good[root + 3] = [0.08, 0.0, 0.0]
    bad = good.copy()
    bad[8] = bad[7] + np.array([0.2, 0.0, 0.0])
    frames = [
        {"hands": [{"label": "Left", "world": good.tolist(), "joint_confidence": [0.9] * 21, "inferred": False}]},
        {"hands": [{"label": "Left", "world": bad.tolist(), "joint_confidence": [0.9] * 8 + [0.1] + [0.9] * 12, "inferred": False}]},
    ]
    enforce_hand_shape(frames)
    fixed = np.asarray(frames[1]["hands"][0]["world"])
    assert abs(np.linalg.norm(fixed[8] - fixed[7]) - 0.02) < 1e-6


def test_low_confidence_joint_moves_toward_neighbors():
    a = _blank()
    b = _blank()
    b[8] = [1.0, 0.0, 0.0]
    frames = [
        {"hands": [{"label": "Left", "world": a.tolist(), "joint_confidence": [1.0] * 21}]},
        {"hands": [{"label": "Left", "world": b.tolist(), "joint_confidence": [1.0] * 8 + [0.0] + [1.0] * 12}]},
    ]
    smooth_by_confidence(frames, window=3)
    moved = np.asarray(frames[1]["hands"][0]["world"])[8, 0]
    assert moved < 0.6


def test_closing_hand_is_a_grasp_and_drift_is_the_wrist_gap():
    frames = []
    for index, curl in enumerate((0.0, 0.04, 0.1)):
        points = _blank()
        _chain(points, curl)
        points[0] = [0.0, 0.0, 0.1 * index]
        hand = {"label": "Right", "world": points.tolist(), "world_fixed": points.tolist()}
        frames.append({"frame": index, "t_sec": index / 30.0, "hands": [hand]})
    annotate_actions(frames, fps=30.0)
    assert frames[1]["hands"][0]["action"] == "grasp"
    frames[0]["hands"][0]["action"] = "release"
    frames[0]["hands"][0]["world_fixed"] = _blank().tolist()
    frames[1]["hands"][0]["action"] = "idle"
    frames[2]["hands"][0]["action"] = "grasp"
    drift = putdown_pickup_drift(frames)
    assert drift["pairs"] == 1
    assert drift["mean_m"] == 0.2


def test_rula_bands():
    assert _rula_wrist(0.0, 0.0) == 1
    assert _rula_wrist(10.0, 0.0) == 2
    assert _rula_wrist(20.0, 20.0) == 4
