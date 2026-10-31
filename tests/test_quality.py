"""TouchTrace Quality Score."""

from __future__ import annotations

import numpy as np

from src.quality import grade_for, touchtrace
from tests.test_scale import canonical_hand, make_frame, make_hand, pose


def _noisy_hand(index: int, **kwargs) -> dict:
    points = canonical_hand()
    jitter = 0.05 * ((index % 3) - 1)
    points = points * (1.0 + jitter)
    return make_hand(points, **kwargs)


def test_grade_thresholds():
    assert grade_for(0.9) == "A"
    assert grade_for(0.85) == "A"
    assert grade_for(0.7) == "B"
    assert grade_for(0.55) == "C"
    assert grade_for(0.4) == "D"


def test_clean_segment_scores_above_noisy_flipped_segment():
    frames = []
    for i in range(6):
        frames.append(
            make_frame(
                i,
                [make_hand(canonical_hand(), action="reach", trust=0.95)],
                pose([0.0, 0.0, 0.0]),
            )
        )
    for i in range(6, 12):
        hand = _noisy_hand(i, action="hold", trust=0.2)
        hand["raw_label"] = "Right"
        frames.append(make_frame(i, [hand], pose([0.0, 0.0, 0.0])))
    segments = [
        {"label": "Left", "action": "reach", "grasp": None, "start_frame": 0, "end_frame": 5,
         "start_sec": 0.0, "end_sec": 0.1667},
        {"label": "Left", "action": "hold", "grasp": "power", "start_frame": 6, "end_frame": 11,
         "start_sec": 0.2, "end_sec": 0.3667},
    ]
    poses = [dict(valid=True, inliers=40)] * 12
    result = touchtrace(frames, segments, poses)
    assert result["overall"] is not None
    assert result["grade"] in {"A", "B", "C", "D"}
    assert set(result["components"]) == {"trust", "bone", "identity", "world", "plausibility"}
    clean, noisy = result["segments"][0], result["segments"][1]
    assert clean["quality"] > noisy["quality"]
    assert clean["grade"] in {"A", "B"}
    assert noisy["quality"] < clean["quality"]
    for i in range(6):
        assert frames[i]["hands"][0]["quality"] is not None
        assert frames[i]["hands"][0]["quality_parts"]["identity"] == 1.0
    assert frames[6]["hands"][0]["quality_parts"]["identity"] == 0.0


def test_implausible_transition_penalized():
    frames = [
        make_frame(0, [make_hand(canonical_hand(), action="grasp", trust=0.9)], pose([0.0, 0.0, 0.0])),
        make_frame(1, [make_hand(canonical_hand(), action="reach", trust=0.9)], pose([0.0, 0.0, 0.0])),
    ]
    segments = [
        {"label": "Left", "action": "grasp", "grasp": "power", "start_frame": 0, "end_frame": 0,
         "start_sec": 0.0, "end_sec": 0.0},
        {"label": "Left", "action": "reach", "grasp": None, "start_frame": 1, "end_frame": 1,
         "start_sec": 0.0333, "end_sec": 0.0333},
    ]
    poses = [dict(valid=True, inliers=40)] * 2
    touchtrace(frames, segments, poses)
    assert frames[0]["hands"][0]["quality_parts"]["plausibility"] == 1.0
    assert frames[1]["hands"][0]["quality_parts"]["plausibility"] == 0.2


def test_world_support_reflects_pose_validity():
    frames = [
        make_frame(0, [make_hand(canonical_hand(), action="reach", trust=0.9)]),
        make_frame(1, [make_hand(canonical_hand(), action="reach", trust=0.9)]),
    ]
    poses = [dict(valid=True, inliers=60), dict(valid=False, inliers=0)]
    touchtrace(frames, [], poses)
    good = frames[0]["hands"][0]["quality_parts"]["world"]
    bad = frames[1]["hands"][0]["quality_parts"]["world"]
    assert good > bad
    assert bad == 0.2


def test_no_hands_gives_null_overall():
    result = touchtrace([make_frame(0, [])], [], None)
    assert result["overall"] is None
    assert result["grade"] is None
