"""Contact-driven physics grounding."""

from __future__ import annotations

import numpy as np

from src.ground import ground_contacts
from tests.test_scale import canonical_hand, make_frame, make_hand, pose


def _contact_clip(n_frames: int = 20, hold_range=range(8, 16)):
    frames = []
    for i in range(n_frames):
        camera = canonical_hand()
        camera[:, 0] -= 0.01 * i
        action = "hold" if i in hold_range else "reach"
        hand = make_hand(camera, label="Right", action=action, camera=camera)
        frames.append(make_frame(i, [hand], pose([0.02 * i, 0.0, 0.0])))
    return frames


def test_grounding_applied_and_anchors_hold():
    frames = _contact_clip()
    result = ground_contacts(frames, fps=30.0)
    assert result["applied"] is True
    assert result["hand_scale"]["scale"] is not None
    assert result["vo_scale"] is not None
    assert len(result["anchors"]) == 1
    anchor = result["anchors"][0]
    assert anchor["start_frame"] == 8
    assert anchor["end_frame"] == 15
    # True world position is static: camera-frame retreat matches scaled VO advance.
    assert np.allclose(anchor["anchor_xyz"], [0.0, 0.0, 0.0], atol=1e-3)
    for i in range(8, 16):
        hand = frames[i]["hands"][0]
        grounded = np.asarray(hand["world_grounded"], dtype=np.float64)
        assert np.allclose(grounded[0], [0.0, 0.0, 0.0], atol=1e-3)


def test_grounding_preserves_hand_shape():
    frames = _contact_clip()
    result = ground_contacts(frames, fps=30.0)
    assert result["applied"] is True
    hand = frames[10]["hands"][0]
    grounded = np.asarray(hand["world_grounded"], dtype=np.float64)
    camera = np.asarray(hand["camera"], dtype=np.float64) * result["hand_scale"]["scale"]
    for child, parent in ((9, 0), (10, 9), (6, 5)):
        original = np.linalg.norm(camera[child] - camera[parent])
        after = np.linalg.norm(grounded[child] - grounded[parent])
        assert abs(original - after) < 1e-6


def test_grounding_skips_cleanly_without_contacts():
    frames = []
    for i in range(10):
        camera = canonical_hand()
        camera[:, 0] -= 0.01 * i
        hand = make_hand(camera, action="reach", camera=camera)
        frames.append(make_frame(i, [hand], pose([0.02 * i, 0.0, 0.0])))
    result = ground_contacts(frames, fps=30.0)
    assert result["applied"] is False
    assert all("world_grounded" not in frame["hands"][0] for frame in frames)


def test_grounding_skips_without_camera():
    frames = [make_frame(i, [make_hand(canonical_hand(), action="hold")]) for i in range(10)]
    result = ground_contacts(frames, fps=30.0)
    assert result["applied"] is False
    assert all("world_grounded" not in frame["hands"][0] for frame in frames)


def test_grounded_wrist_series_exported():
    frames = _contact_clip()
    result = ground_contacts(frames, fps=30.0)
    assert result["applied"] is True
    assert (10, "Right") in result["wrists"]
    assert np.allclose(result["wrists"][(10, "Right")], [0.0, 0.0, 0.0], atol=1e-3)
