"""Ground-truth evaluation for hand tracking JSONL (TouchTrace / PS4)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from src.hands import landmarks_bbox


def load_jsonl(path: Path) -> dict[int, dict]:
    """Index frames by frame number."""
    out: dict[int, dict] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            out[int(row["frame"])] = row
    return out


def _wrist_xy(hand: dict) -> np.ndarray | None:
    image = hand.get("image")
    if not image or len(image) < 1:
        return None
    pt = np.asarray(image[0], dtype=np.float64).reshape(-1)
    if pt.size < 2:
        return None
    return pt[:2]


def _landmark_xy(hand: dict) -> np.ndarray | None:
    image = hand.get("image")
    if not image or len(image) != 21:
        return None
    arr = np.asarray(image, dtype=np.float64).reshape(-1, 3)
    return arr[:, :2]


def _paired_landmarks(pred: dict, gt: dict) -> tuple[np.ndarray, np.ndarray] | None:
    """Align prediction and GT landmarks, supporting partial (indexed) GT.

    Full 21-point GT needs no `indices`. Partial GT (e.g. wrist + five
    fingertips from the 6-point annotation mode) supplies `indices` naming
    which MediaPipe joints the points correspond to.
    """
    pg = _landmark_xy(pred)
    image = gt.get("image") or []
    if pg is None or not image:
        return None
    gg = np.asarray(image, dtype=np.float64).reshape(-1, 3)[:, :2]
    indices = gt.get("indices")
    if indices is None:
        if len(gg) != 21:
            return None
        indices = list(range(21))
    if len(indices) != len(gg) or any(int(i) >= len(pg) for i in indices):
        return None
    return pg[np.asarray(indices, dtype=int)], gg


def match_hands_spatial(
    predictions: list[dict],
    ground_truth: list[dict],
    *,
    match_dist: float,
) -> tuple[list[tuple[int, int]], list[int], list[int]]:
    """Greedy match by wrist distance in normalized image coordinates."""
    pairs: list[tuple[float, int, int]] = []
    for pi, pred in enumerate(predictions):
        pw = _wrist_xy(pred)
        if pw is None:
            continue
        pb = pred.get("bbox")
        if pb is None:
            pb = landmarks_bbox(pred["image"])
        for gi, gt in enumerate(ground_truth):
            gw = _wrist_xy(gt)
            if gw is None:
                gb = gt.get("bbox")
                if gb is None and gt.get("image"):
                    gb = landmarks_bbox(gt["image"])
                if gb is not None:
                    gw = np.array([(gb[0] + gb[2]) / 2, (gb[1] + gb[3]) / 2])
                else:
                    continue
            dist = float(np.linalg.norm(pw - gw))
            if dist <= match_dist:
                pairs.append((dist, pi, gi))
    pairs.sort()
    used_p: set[int] = set()
    used_g: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _d, pi, gi in pairs:
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi)
        used_g.add(gi)
        matches.append((pi, gi))
    unmatched_p = [i for i in range(len(predictions)) if i not in used_p]
    unmatched_g = [i for i in range(len(ground_truth)) if i not in used_g]
    return matches, unmatched_p, unmatched_g


def _mean_landmark_error(pred: dict, gt: dict) -> float | None:
    paired = _paired_landmarks(pred, gt)
    if paired is None:
        return None
    pg, gg = paired
    return float(np.linalg.norm(pg - gg, axis=1).mean())


def _pck(pred: dict, gt: dict, threshold: float) -> float | None:
    paired = _paired_landmarks(pred, gt)
    if paired is None:
        return None
    pg, gg = paired
    dists = np.linalg.norm(pg - gg, axis=1)
    return float(np.mean(dists <= threshold))


def evaluate_ground_truth(
    predictions_by_frame: dict[int, dict],
    labels_by_frame: dict[int, dict],
    *,
    match_dist: float = 0.08,
    pck_threshold: float = 0.05,
) -> dict[str, Any]:
    """Compare predictions to labelled frames only."""
    labelled_frames = sorted(labels_by_frame.keys())
    if not labelled_frames:
        return {
            "status": "no_labelled_frames",
            "message": "No ground-truth annotations supplied; accuracy cannot be measured.",
            "labelled_frames": 0,
        }

    tp = fp = fn = 0
    handedness_correct = 0
    handedness_total = 0
    landmark_errors: list[float] = []
    pck_scores: list[float] = []
    tracking: dict[str, list[tuple[int, str]]] = {}

    for frame in labelled_frames:
        gt_frame = labels_by_frame[frame]
        gt_hands = gt_frame.get("hands") or []
        pred_frame = predictions_by_frame.get(frame)
        pred_hands = (pred_frame or {}).get("hands") or []

        if not gt_hands:
            fp += len(pred_hands)
            continue

        matches, unmatched_p, unmatched_g = match_hands_spatial(
            pred_hands, gt_hands, match_dist=match_dist
        )
        tp += len(matches)
        fp += len(unmatched_p)
        fn += len(unmatched_g)

        for pi, gi in matches:
            pred = pred_hands[pi]
            gt = gt_hands[gi]
            if pred.get("label") == gt.get("label"):
                handedness_correct += 1
            handedness_total += 1

            err = _mean_landmark_error(pred, gt)
            if err is not None:
                landmark_errors.append(err)
            pck = _pck(pred, gt, pck_threshold)
            if pck is not None:
                pck_scores.append(pck)

            gt_label = str(gt.get("label", f"hand_{gi}"))
            tracking.setdefault(gt_label, []).append((frame, str(pred.get("label", ""))))

    precision = tp / (tp + fp) if (tp + fp) > 0 else None
    recall = tp / (tp + fn) if (tp + fn) > 0 else None
    detection_rate = tp / (tp + fn) if (tp + fn) > 0 else None

    identity_switches = 0
    for _key, seq in tracking.items():
        labels = [lab for _f, lab in seq]
        for a, b in zip(labels, labels[1:]):
            if a and b and a != b:
                identity_switches += 1

    return {
        "status": "ok",
        "labelled_frames": len(labelled_frames),
        "detection": {
            "true_positives": tp,
            "false_positives": fp,
            "false_negatives": fn,
            "precision": round(precision, 4) if precision is not None else None,
            "recall": round(recall, 4) if recall is not None else None,
            "detection_rate": round(detection_rate, 4) if detection_rate is not None else None,
            "note": "Spatial wrist match; labels ignored for TP/FP/FN.",
        },
        "landmarks": {
            "mean_error_norm": round(float(np.mean(landmark_errors)), 6) if landmark_errors else None,
            "pck": round(float(np.mean(pck_scores)), 4) if pck_scores else None,
            "pck_threshold_norm": pck_threshold,
            "coordinate_note": "Normalized image x,y in [0,1] relative to frame width/height.",
            "pairs_with_full_landmarks": len(landmark_errors),
        },
        "handedness": {
            "accuracy": round(handedness_correct / handedness_total, 4) if handedness_total else None,
            "correct": handedness_correct,
            "spatially_matched_pairs": handedness_total,
            "note": "Among spatial matches only; separate from detection precision/recall.",
        },
        "tracking": {
            "identity_label_switches": identity_switches,
            "note": "Counts prediction label changes for the same spatially matched GT hand across labelled frames.",
        },
        "config": {"match_dist_norm": match_dist, "pck_threshold_norm": pck_threshold},
    }


def merge_operational_report(operational: dict | None) -> dict[str, Any]:
    if not operational:
        return {}
    return {"operational": operational}
