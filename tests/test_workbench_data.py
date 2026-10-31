import json
from pathlib import Path

import numpy as np

from src.workbench_data import (
    clip_paths,
    detection_codes,
    discover_clips,
    hand_rows,
    max_frames_for_seconds,
    mesh_surfaces,
    safe_stem,
    save_upload,
)


def test_discover_and_paths(tmp_path: Path):
    (tmp_path / "clip_sidebyside.mp4").write_bytes(b"x")
    (tmp_path / "clip_report.json").write_text("{}", encoding="utf-8")
    (tmp_path / "clip.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "other_mesh.jsonl").write_text("", encoding="utf-8")
    assert discover_clips(tmp_path) == ["clip"]
    paths = clip_paths(tmp_path, "clip")
    assert paths["video"].name == "clip_sidebyside.mp4"
    assert paths["mesh_npz"] is None


def test_detection_codes_and_hand_rows():
    rows = [
        {"hands": [{"label": "Left"}]},
        {"hands": [{"label": "Left"}, {"label": "Right"}]},
        {"hands": []},
    ]
    assert detection_codes(rows) == [1, 3, 0]
    frame = {
        "hands": [
            {
                "label": "Right",
                "score": 0.5,
                "tracked": True,
                "interpolated": False,
                "world": [[0, 0, 0]] * 21,
                "reproj_error_px": 2.5,
                "joint_confidence": [0.8] * 21,
                "action": "grasp",
            }
        ]
    }
    table = hand_rows(frame, "world")
    assert table[0]["label"] == "Right"
    assert table[0]["joints"] == 21
    assert table[0]["mean_trust"] == 0.8
    assert table[0]["action"] == "grasp"
    assert json.dumps(table)


def test_mesh_in_a_result_folder_is_visible(tmp_path: Path):
    folder = tmp_path / "mesh50"
    folder.mkdir()
    np.savez(
        folder / "clip_mesh.npz",
        vertices=np.zeros((1, 4, 3), np.float32),
        frame_ids=np.array([3], np.int32),
        labels=np.array([1], np.int8),
        faces=np.array([[0, 1, 2]], np.int32),
    )
    assert "clip" in discover_clips(tmp_path)
    assert clip_paths(tmp_path, "clip")["mesh_npz"].parent == folder
    surfaces = mesh_surfaces(
        {"vertices": np.zeros((1, 4, 3)), "frame_ids": np.array([3]), "labels": np.array([1]), "faces": np.zeros((1, 3))},
        3,
    )
    assert surfaces[0]["label"] == "Right"
    assert mesh_surfaces(
        {"vertices": np.zeros((1, 4, 3)), "frame_ids": np.array([3]), "labels": np.array([1]), "faces": np.zeros((1, 3))},
        1,
    ) == []


def test_upload_name_and_duration(tmp_path: Path):
    assert safe_stem("../../My Clip!.mp4") == "My_Clip"
    assert safe_stem("...") == "clip"
    dest = save_upload(tmp_path, "My Clip.mov", b"video")
    assert dest.name == "My_Clip.mp4"
    assert dest.read_bytes() == b"video"
    assert max_frames_for_seconds(30, 50) == 1500
    assert max_frames_for_seconds(29.97, 0) is None
