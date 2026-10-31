"""World-frame estimation, trajectory smoothing, and simulator export tests."""

from __future__ import annotations

import csv

import numpy as np
from scipy.spatial.transform import Rotation

from src.export_sim import export_world_sim, finger_curls, write_world_csv
from src.mesh import merge_frame
from src.world import (
    CameraMotionEstimator,
    _relative_pose,
    _rotation_from_homography,
    attach_world,
    build_world_trajectories,
    smooth_world_trajectories,
)


def _project(points, k):
    uv = (k @ points.T).T
    return uv[:, :2] / uv[:, 2:3]


def _sample(frame, wrist, quat=(0.0, 0.0, 0.0, 1.0)):
    return {
        "frame": frame,
        "t_sec": round(frame / 30.0, 4),
        "wrist_xyz": list(wrist),
        "quat_xyzw": None if quat is None else list(quat),
        "tracked": True,
        "interpolated": False,
        "smoothed": False,
    }


def test_relative_pose_recovers_synthetic_motion():
    rng = np.random.default_rng(0)
    k = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    pts = np.column_stack([rng.uniform(-1, 1, 120), rng.uniform(-1, 1, 120), rng.uniform(2, 6, 120)])
    rot = Rotation.from_euler("y", 5, degrees=True).as_matrix()
    tvec = np.array([0.05, 0.02, 0.1])
    prev = _project(pts, k)
    curr = _project((rot @ pts.T).T + tvec, k)
    result = _relative_pose(prev, curr, k)
    assert result is not None
    r_rec, t_rec, inliers = result
    assert inliers >= 100
    assert Rotation.from_matrix(r_rec @ rot.T).magnitude() < 0.01
    cosine = float(np.dot(t_rec / np.linalg.norm(t_rec), tvec / np.linalg.norm(tvec)))
    assert cosine > 0.99


def test_relative_pose_returns_none_on_too_few_points():
    k = np.eye(3)
    assert _relative_pose(np.zeros((4, 2)), np.zeros((4, 2)), k) is None


def test_homography_fallback_recovers_pure_rotation():
    rng = np.random.default_rng(6)
    k = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
    pts = np.column_stack([rng.uniform(-1, 1, 100), rng.uniform(-1, 1, 100), rng.uniform(2, 6, 100)])
    rot = Rotation.from_euler("xyz", [2, 5, -3], degrees=True).as_matrix()
    prev = _project(pts, k)
    curr = _project((rot @ pts.T).T, k)
    result = _rotation_from_homography(prev, curr, k)
    assert result is not None
    r_rec, inliers = result
    assert inliers >= 90
    assert Rotation.from_matrix(r_rec @ rot.T).magnitude() < 0.02


def test_homography_fallback_returns_none_on_few_points():
    k = np.eye(3)
    assert _rotation_from_homography(np.zeros((5, 2)), np.zeros((5, 2)), k) is None


def test_apply_relative_composes_inverse_transform():
    est = CameraMotionEstimator(640, 480)
    rot = Rotation.from_euler("y", 10, degrees=True).as_matrix()
    tvec = np.array([0.1, 0.0, 0.0])
    est._apply_relative(rot, tvec)
    assert np.allclose(est._rot, rot.T)
    assert np.allclose(est._t, -rot.T @ tvec)


def test_blank_frames_anchor_then_carry_forward():
    est = CameraMotionEstimator(64, 48)
    blank = np.zeros((48, 64, 3), dtype=np.uint8)
    est.step(blank)
    est.step(blank)
    poses = est.trajectory()
    assert len(poses) == 2
    assert poses[0]["valid"] is True
    assert np.allclose(poses[0]["rotation"], np.eye(3))
    assert poses[1]["valid"] is False
    assert np.allclose(poses[1]["translation"], poses[0]["translation"])
    assert np.allclose(poses[1]["rotation"], poses[0]["rotation"])


def test_background_mask_excludes_hand_boxes():
    est = CameraMotionEstimator(200, 100)
    mask = est._background_mask([(0.4, 0.4, 0.6, 0.6)])
    assert mask[50, 100] == 0
    assert mask[45, 85] == 0
    assert mask[10, 10] == 255
    assert mask[50, 20] == 255


def test_attach_world_transforms_camera_joints():
    rot = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    poses = [{"rotation": rot, "translation": np.array([1.0, 0, 0]), "valid": True, "inliers": 50}]
    camera = np.zeros((21, 3))
    camera[0] = [1.0, 0, 0]
    frames = [
        {
            "frame": 0,
            "t_sec": 0.0,
            "hands": [
                {"label": "Left", "world": camera.tolist(), "camera": camera.tolist()},
                {"label": "Right", "world": camera.tolist(), "camera": None},
            ],
        }
    ]
    attach_world(frames, poses)
    left, right = frames[0]["hands"]
    assert np.allclose(left["world_fixed"][0], [1.0, 1.0, 0.0], atol=1e-9)
    assert right["world_fixed"] is None
    assert frames[0]["camera_pose_world"]["valid"] is True


def test_attach_world_missing_pose():
    frames = [{"frame": 0, "t_sec": 0.0, "hands": [{"label": "Left", "world": [], "camera": None}]}]
    attach_world(frames, [])
    assert frames[0]["camera_pose_world"] is None
    assert frames[0]["hands"][0]["world_fixed"] is None


def test_smoothing_reduces_position_noise():
    rng = np.random.default_rng(1)
    frames_idx = np.arange(30)
    noisy = np.column_stack([0.01 * frames_idx, np.zeros(30), np.zeros(30)]) + rng.normal(0, 0.005, (30, 3))
    samples = [_sample(int(f), wrist) for f, wrist in zip(frames_idx, noisy)]
    trajectories = {"Left": samples}
    smooth_world_trajectories(trajectories, window=11)
    out = trajectories["Left"]
    assert len(out) == 30
    assert all(s["smoothed"] for s in out)
    smoothed = np.asarray([s["wrist_xyz"] for s in out])
    ideal = np.column_stack([0.01 * frames_idx, np.zeros(30), np.zeros(30)])
    assert (smoothed - ideal).std() < (noisy - ideal).std()


def test_smoothing_keeps_quaternion_signs_continuous():
    angles = np.linspace(0.0, 1.0, 30)
    quats = [Rotation.from_euler("z", a).as_quat() for a in angles]
    rng = np.random.default_rng(2)
    flipped = [(-q if rng.random() < 0.5 else q).tolist() for q in quats]
    samples = [_sample(f, [0.0, 0, 0], quat=q) for f, q in enumerate(flipped)]
    trajectories = {"Left": samples}
    smooth_world_trajectories(trajectories, window=11)
    out = np.asarray([s["quat_xyzw"] for s in trajectories["Left"]])
    dots = np.sum(out[1:] * out[:-1], axis=1)
    assert np.all(dots > 0)
    assert np.allclose(np.linalg.norm(out, axis=1), 1.0)


def test_short_segment_passes_through():
    samples = [_sample(f, [0.1 * f, 0.0, 0.0]) for f in range(3)]
    before = [list(s["wrist_xyz"]) for s in samples]
    trajectories = {"Left": samples}
    smooth_world_trajectories(trajectories, window=11)
    out = trajectories["Left"]
    assert [s["wrist_xyz"] for s in out] == before
    assert not any(s["smoothed"] for s in out)


def test_small_gap_is_filled_and_large_gap_splits():
    samples = [_sample(f, [0.1 * f, 0.0, 0.0]) for f in (0, 1, 2, 5, 6)]
    trajectories = {"Left": samples}
    smooth_world_trajectories(trajectories, window=7)
    out = trajectories["Left"]
    assert [s["frame"] for s in out] == [0, 1, 2, 3, 4, 5, 6]
    filled = [s for s in out if s["frame"] in (3, 4)]
    assert all(s["interpolated"] and not s["tracked"] for s in filled)
    assert np.allclose([s["wrist_xyz"][0] for s in out], [0.1 * f for f in range(7)], atol=1e-6)

    split = {"Left": [_sample(f, [0.0, 0, 0]) for f in (0, 1, 10, 11)]}
    smooth_world_trajectories(split, window=7)
    assert [s["frame"] for s in split["Left"]] == [0, 1, 10, 11]


def test_build_world_trajectories_skips_missing_world():
    rng = np.random.default_rng(3)
    pts = rng.random((21, 3))
    frames = [
        {"frame": 0, "t_sec": 0.0, "hands": [{"label": "Left", "world_fixed": pts.tolist()}]},
        {"frame": 1, "t_sec": 0.0333, "hands": [{"label": "Left", "world_fixed": None}]},
    ]
    trajectories = build_world_trajectories(frames)
    assert len(trajectories["Left"]) == 1
    sample = trajectories["Left"][0]
    assert np.allclose(sample["wrist_xyz"], pts[0])
    assert sample["quat_xyzw"] is not None


def test_finger_curls_are_rigid_invariant():
    rng = np.random.default_rng(4)
    pts = rng.random((21, 3))
    rot = Rotation.random(random_state=4).as_matrix()
    moved = pts @ rot.T + np.array([0.5, -0.2, 1.0])
    before = finger_curls(pts)
    after = finger_curls(moved)
    for name in before:
        assert abs(before[name] - after[name]) < 1e-9


def test_world_export_schema_and_csv(tmp_path):
    rng = np.random.default_rng(5)
    pts = rng.random((21, 3))
    frames = [
        {
            "frame": 0,
            "t_sec": 0.0,
            "hands": [{"label": "Left", "world": pts.tolist(), "camera": pts.tolist(), "tracked": True}],
        }
    ]
    trajectories = {"Left": [_sample(0, [0.1, 0.2, 0.3])]}
    trajectories["Left"][0]["smoothed"] = True
    doc = export_world_sim(trajectories, frames, fps=30.0, source="clip.mp4")
    assert doc["coordinate_frame"]["name"] == "camera_at_frame_0"
    hand = doc["frames"][0]["hands"][0]
    assert hand["label"] == "Left"
    assert hand["wrist_xyz"] == [0.1, 0.2, 0.3]
    assert hand["wrist_quat_xyzw"] == [0.0, 0.0, 0.0, 1.0]
    assert hand["smoothed"] is True
    assert hand["finger_curls_rad"] is not None

    path = tmp_path / "world.csv"
    write_world_csv(path, doc)
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["frame"] == "0"
    assert rows[0]["label"] == "Left"
    assert rows[0]["tracked"] == "1"
    assert rows[0]["curl_index"] != ""


def test_merge_frame_preserves_world_fixed():
    base = {
        "frame": 0,
        "t_sec": 0.0,
        "hands": [{"label": "Left", "world": [], "image": [], "camera": None, "world_fixed": [[0, 0, 0]]}],
    }
    mesh = {
        "frame": 0,
        "t_sec": 0.0,
        "hands": [{"label": "Left", "source": "wilor", "world": [], "image": [], "camera": None, "mesh_index": 3}],
    }
    merged = merge_frame(base, mesh)
    hand = merged["hands"][0]
    assert hand["world_fixed"] == [[0, 0, 0]]
    assert hand["mesh"]["mesh_index"] == 3
