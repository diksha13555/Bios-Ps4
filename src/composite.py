"""Detection benchmark: real hand pixels composited over synthetic backgrounds.

A real hand crop is pasted along a scripted trajectory with known scale, so the
wrist position and bounding box are exact ground truth (not model output).
Labels carry only wrist + bbox: detection precision/recall, wrist localization
error, and identity stability are measurable; landmark PCK intentionally is
not, because 21-point labels would have to come from a model.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from src.hands import landmarks_bbox
from src.synth import FPS, HEIGHT, WIDTH, render_background


def load_hand_crop(video_path: Path, predictions_path: Path, min_score: float = 0.75) -> dict:
    """Best-scoring real hand crop plus its wrist offset inside the crop."""
    rows = [json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    best = None
    for row in rows:
        real_hands = [h for h in row["hands"] if not h.get("inferred") and float(h.get("score") or 0.0) >= min_score]
        # Only single-hand frames: a second hand inside the crop would be pasted
        # too, and every detection of it would count as a false positive.
        if len(real_hands) != 1:
            continue
        hand = real_hands[0]
        if best is None or hand["score"] > best[0]["score"]:
            best = (hand, row["frame"])
    if best is None:
        raise RuntimeError("No confidently detected single-hand frame found for the composite benchmark.")
    hand, frame_index = best
    cap = cv2.VideoCapture(str(video_path))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read frame {frame_index} of {video_path}")
    x1, y1, x2, y2 = landmarks_bbox(hand["image"], pad=0.02)
    h, w = frame.shape[:2]
    px1, py1 = int(max(0.0, x1) * w), int(max(0.0, y1) * h)
    px2, py2 = int(min(1.0, x2) * w), int(min(1.0, y2) * h)
    crop = frame[py1:py2, px1:px2].copy()
    wrist = hand["image"][0]
    return {
        "crop": crop,
        "label": hand["label"],
        "wrist_offset": np.array([
            float(wrist[0]) * w - px1,
            float(wrist[1]) * h - py1,
        ]),
    }


def generate_composite(video_path: Path, predictions_path: Path, output_dir: Path, n_frames: int = 240) -> dict:
    """Render the composite clip with exact wrist/bbox ground-truth labels."""
    output_dir.mkdir(parents=True, exist_ok=True)
    source = load_hand_crop(video_path, predictions_path)
    crop = source["crop"]
    ch, cw = crop.shape[:2]
    background = render_background()
    video_path_out = output_dir / "composite.mp4"
    labels_path = output_dir / "composite_labels.jsonl"
    writer = cv2.VideoWriter(str(video_path_out), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (WIDTH, HEIGHT))
    labels = []
    for index in range(n_frames):
        frame = background.copy()
        hands = []
        # Hand is off-screen for the first and last 20 frames: negative examples.
        if 20 <= index < n_frames - 20:
            phase = (index - 20) / (n_frames - 40)
            scale = 0.75 + 0.5 * (0.5 + 0.5 * np.sin(phase * 2 * np.pi))
            center = np.array([
                WIDTH * (0.15 + 0.7 * phase),
                HEIGHT * (0.42 + 0.12 * np.sin(phase * 3 * np.pi)),
            ])
            tw, th = max(8, int(cw * scale)), max(8, int(ch * scale))
            resized = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)
            x1 = int(center[0] - tw / 2)
            y1 = int(center[1] - th / 2)
            sx1, sy1 = max(0, x1), max(0, y1)
            sx2, sy2 = min(WIDTH, x1 + tw), min(HEIGHT, y1 + th)
            if sx2 > sx1 and sy2 > sy1:
                patch = resized[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1]
                # Feather the patch border so the composite has no hard rectangle seam.
                mask = np.ones(patch.shape[:2], dtype=np.float32)
                edge = max(2, int(0.06 * min(patch.shape[:2])))
                ramp = np.linspace(0.0, 1.0, edge, dtype=np.float32)
                mask[:edge, :] *= ramp[:, None]
                mask[-edge:, :] *= ramp[::-1][:, None]
                mask[:, :edge] *= ramp[None, :]
                mask[:, -edge:] *= ramp[::-1][None, :]
                roi = frame[sy1:sy2, sx1:sx2].astype(np.float32)
                blended = roi * (1 - mask[..., None]) + patch.astype(np.float32) * mask[..., None]
                frame[sy1:sy2, sx1:sx2] = np.clip(blended, 0, 255).astype(np.uint8)
                wrist = center + (source["wrist_offset"] - np.array([cw / 2, ch / 2])) * scale
                hands.append({
                    "label": source["label"],
                    "image": [[round(float(wrist[0]) / WIDTH, 5), round(float(wrist[1]) / HEIGHT, 5), 0.0]],
                    "bbox": [round(sx1 / WIDTH, 5), round(sy1 / HEIGHT, 5), round(sx2 / WIDTH, 5), round(sy2 / HEIGHT, 5)],
                })
        writer.write(frame)
        labels.append({"frame": index, "hands": hands})
    writer.release()
    with labels_path.open("w", encoding="utf-8") as handle:
        for row in labels:
            handle.write(json.dumps(row) + "\n")
    return {"video": video_path_out, "labels": labels_path}


def wrist_localization(predictions_path: Path, labels_path: Path) -> dict:
    """Mean normalized wrist error on frames where both prediction and GT exist."""
    from src.eval_gt import load_jsonl, match_hands_spatial

    predictions = load_jsonl(predictions_path)
    labels = load_jsonl(labels_path)
    errors = []
    for index, label in labels.items():
        pred = predictions.get(index)
        if pred is None:
            continue
        matches, _, _ = match_hands_spatial(pred["hands"], label["hands"], match_dist=0.08)
        for pi, gi in matches:
            pw = np.asarray(pred["hands"][pi]["image"][0][:2], dtype=np.float64)
            gw = np.asarray(label["hands"][gi]["image"][0][:2], dtype=np.float64)
            errors.append(float(np.linalg.norm(pw - gw)))
    if not errors:
        return {"pairs": 0, "mean_wrist_error_norm": None}
    return {
        "pairs": len(errors),
        "mean_wrist_error_norm": round(float(np.mean(errors)), 5),
        "note": "Normalized image units; GT wrist comes from the compositing transform, not a model.",
    }
