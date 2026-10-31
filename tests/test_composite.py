"""Composite benchmark and partial ground-truth evaluation."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.composite import generate_composite, load_hand_crop, wrist_localization
from src.eval_gt import evaluate_ground_truth, load_jsonl


def _write_video(path: Path, n_frames: int = 3, size=(64, 48)) -> None:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, size)
    for index in range(n_frames):
        frame = np.full((size[1], size[0], 3), 40 + 30 * index, dtype=np.uint8)
        writer.write(frame)
    writer.release()


def _predictions(path: Path) -> None:
    rows = [
        # Two-hand frame: must not be selected as the crop source.
        {"frame": 0, "hands": [
            {"label": "Left", "score": 0.99, "image": [[0.5, 0.5, 0.0]] * 21},
            {"label": "Right", "score": 0.9, "image": [[0.4, 0.4, 0.0]] * 21},
        ]},
        {"frame": 1, "hands": []},
        {"frame": 2, "hands": [
            {"label": "Right", "score": 0.9, "image": [[0.5, 0.5, 0.0]] * 21},
        ]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")


def test_crop_source_is_single_hand_frame(tmp_path):
    video = tmp_path / "clip.mp4"
    preds = tmp_path / "clip.jsonl"
    _write_video(video)
    _predictions(preds)
    source = load_hand_crop(video, preds)
    assert source["label"] == "Right"
    assert source["crop"].size > 0


def test_composite_labels_have_exact_wrist_and_negative_frames(tmp_path):
    video = tmp_path / "clip.mp4"
    preds = tmp_path / "clip.jsonl"
    _write_video(video)
    _predictions(preds)
    out = generate_composite(video, preds, tmp_path / "out", n_frames=50)
    labels = load_jsonl(out["labels"])
    assert labels[0]["hands"] == []
    assert labels[49]["hands"] == []
    hand = labels[25]["hands"][0]
    assert hand["label"] == "Right"
    wx, wy, _ = hand["image"][0]
    bx1, by1, bx2, by2 = hand["bbox"]
    assert bx1 <= wx <= bx2 and by1 <= wy <= by2
    assert 0.0 <= wx <= 1.0 and 0.0 <= wy <= 1.0


def test_partial_landmark_pck_uses_indices(tmp_path):
    pred_points = [[0.1 * (i % 4), 0.1 * (i % 3), 0.0] for i in range(21)]
    gt_indices = [0, 4, 8, 12, 16, 20]
    gt_points = [pred_points[i] for i in gt_indices]
    gt_points[2] = [0.9, 0.9, 0.0]  # one of six points far away
    predictions = {0: {"frame": 0, "hands": [{"label": "Right", "image": pred_points}]}}
    labels = {0: {"frame": 0, "hands": [{"label": "Right", "image": gt_points, "indices": gt_indices}]}}
    result = evaluate_ground_truth(predictions, labels)
    assert result["status"] == "ok"
    assert abs(result["landmarks"]["pck"] - 5 / 6) < 1e-3
    assert result["landmarks"]["mean_error_norm"] is not None


def test_wrist_localization_uses_gt_transform(tmp_path):
    predictions = tmp_path / "pred.jsonl"
    labels = tmp_path / "labels.jsonl"
    pred_row = {"frame": 0, "hands": [{"label": "Right", "image": [[0.45, 0.5, 0.0]] + [[0.0, 0.0, 0.0]] * 20}]}
    label_row = {"frame": 0, "hands": [{"label": "Right", "image": [[0.4, 0.5, 0.0]], "bbox": [0.3, 0.4, 0.5, 0.6]}]}
    predictions.write_text(json.dumps(pred_row), encoding="utf-8")
    labels.write_text(json.dumps(label_row), encoding="utf-8")
    result = wrist_localization(predictions, labels)
    assert result["pairs"] == 1
    assert abs(result["mean_wrist_error_norm"] - 0.05) < 1e-4
