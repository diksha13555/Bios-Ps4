import json
from pathlib import Path

import numpy as np

from src.mesh import merge_frame, merge_jsonl, prediction_to_hand, save_mesh_npz


def _pred(is_right, xyz):
    joints = np.zeros((1, 21, 3), dtype=np.float32)
    joints[0, :, :] = xyz
    pixels = np.zeros((1, 21, 2), dtype=np.float32)
    pixels[0, :, 0] = 100
    pixels[0, :, 1] = 50
    return {
        "is_right": is_right,
        "wilor_preds": {
            "pred_keypoints_3d": joints,
            "pred_keypoints_2d": pixels,
            "pred_vertices": np.ones((1, 4, 3), dtype=np.float32),
            "pred_cam_t_full": np.array([[0.0, 0.0, 2.0]], dtype=np.float32),
        },
    }


def test_prediction_matches_hand_schema():
    hand = prediction_to_hand(_pred(0, [0.1, 0.2, 0.3]), width=200, height=100, mesh_index=3)
    assert hand["label"] == "Left"
    assert len(hand["world"]) == 21
    assert hand["image"][0][0] == 0.5
    assert hand["image"][0][1] == 0.5
    assert hand["camera"][0][2] == 2.3
    assert hand["mesh_index"] == 3
    assert hand["source"] == "wilor"


def test_merge_attaches_mesh_and_keeps_mediapipe_world():
    base = {"frame": 1, "t_sec": 0.033, "hands": [{"label": "Left", "world": [[0, 0, 0]]}]}
    mesh_hand = prediction_to_hand(_pred(0, [1, 0, 0]), 200, 100, 0)
    mesh_hand.pop("vertices")
    mesh_hand.pop("cam_t")
    merged = merge_frame(base, {"frame": 1, "t_sec": 0.033, "hands": [mesh_hand]})
    assert merged["hands"][0]["world"] == [[0, 0, 0]]
    assert merged["hands"][0]["mesh"]["source"] == "wilor"
    assert merged["hands"][0]["mesh"]["world"][0][0] == 1.0


def test_merge_appends_hand_mediapipe_missed():
    base = {"frame": 0, "t_sec": 0.0, "hands": []}
    mesh_hand = prediction_to_hand(_pred(1, [0, 0, 0]), 200, 100, 1)
    mesh_hand.pop("vertices")
    mesh_hand.pop("cam_t")
    merged = merge_frame(base, {"frame": 0, "t_sec": 0.0, "hands": [mesh_hand]})
    assert merged["hands"][0]["label"] == "Right"
    assert merged["hands"][0]["source"] == "wilor"


def test_merge_jsonl_and_npz(tmp_path: Path):
    base = tmp_path / "clip.jsonl"
    mesh = tmp_path / "clip_mesh.jsonl"
    out = tmp_path / "clip_merged.jsonl"
    base.write_text(json.dumps({"frame": 0, "t_sec": 0.0, "hands": [{"label": "Right", "world": [[0, 0, 0]]}]}) + "\n")
    hand = prediction_to_hand(_pred(1, [0.2, 0, 0]), 200, 100, 0)
    public = {k: v for k, v in hand.items() if k not in {"vertices", "cam_t"}}
    mesh.write_text(json.dumps({"frame": 0, "t_sec": 0.0, "hands": [public]}) + "\n")
    assert merge_jsonl(base, mesh, out) == 1
    row = json.loads(out.read_text().strip())
    assert "mesh" in row["hands"][0]
    npz = tmp_path / "mesh.npz"
    save_mesh_npz(
        npz,
        [{"frame": 0, "label": "Right", "world": hand["world"], "vertices": hand["vertices"], "cam_t": hand["cam_t"]}],
        faces=np.array([[0, 1, 2]], dtype=np.int32),
    )
    loaded = np.load(npz)
    assert loaded["vertices"].shape[0] == 1
    assert loaded["faces"].shape == (1, 3)
