"""TouchTrace Quality Score: how much should you trust this reconstruction?

Five signals combine into one 0-1 score per hand-frame: joint trust, bone-length
stability within the action segment, Left/Right identity consistency, world-frame
support (visual-odometry validity and inliers), and action plausibility.
Segments and the whole clip get A/B/C/D grades. Heuristic, not ground truth.
"""

from __future__ import annotations

import numpy as np

from src.geometry import PARENT

WEIGHTS = {"trust": 0.30, "bone": 0.20, "identity": 0.15, "world": 0.20, "plausibility": 0.15}
GRADE_THRESHOLDS = ((0.85, "A"), (0.70, "B"), (0.55, "C"))
IMPLAUSIBLE_TRANSITIONS = frozenset({("grasp", "reach"), ("grasp", "idle"), ("hold", "reach")})
CURL_JUMP_RAD = 1.5


def _bone_cv(series: list[np.ndarray]) -> float:
    values: dict[int, list[float]] = {}
    for points in series:
        for child, parent in PARENT.items():
            length = float(np.linalg.norm(points[child] - points[parent]))
            if length > 1e-6:
                values.setdefault(child, []).append(length)
    cvs = [
        float(np.std(lengths) / np.mean(lengths))
        for lengths in values.values()
        if len(lengths) >= 3 and float(np.mean(lengths)) > 1e-6
    ]
    return float(np.mean(cvs)) if cvs else 0.0


def grade_for(score: float) -> str:
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "D"


def touchtrace(frames: list[dict], segments: list[dict], poses: list[dict] | None = None) -> dict:
    """Attach hand['quality'] per frame and return the clip-level summary."""
    pose_support: dict[int, tuple[bool, int]] = {}
    if poses:
        for i, pose in enumerate(poses):
            pose_support[i] = (bool(pose.get("valid")), int(pose.get("inliers") or 0))

    segment_cv: dict[tuple[str, int, int], float] = {}
    frame_cv: dict[tuple[str, int], float] = {}
    for seg in segments:
        key = (seg["label"], seg["start_frame"], seg["end_frame"])
        series = []
        for frame in frames:
            if not seg["start_frame"] <= frame["frame"] <= seg["end_frame"]:
                continue
            for hand in frame["hands"]:
                if hand["label"] != seg["label"]:
                    continue
                points = np.asarray(hand.get("world") or [], dtype=np.float64)
                if points.ndim == 2 and points.shape[0] >= 21:
                    series.append(points)
        segment_cv[key] = _bone_cv(series) if series else 0.15
        for f in range(seg["start_frame"], seg["end_frame"] + 1):
            frame_cv[(seg["label"], f)] = segment_cv[key]

    totals: dict[str, list[float]] = {key: [] for key in WEIGHTS}
    all_scores: list[float] = []
    prev_state: dict[str, tuple[str, float]] = {}
    for index, frame in enumerate(frames):
        pose_valid, inliers = pose_support.get(index, (False, 0))
        if not poses:
            world_score = 0.5
        elif pose_valid:
            world_score = 0.4 + 0.6 * min(1.0, inliers / 30.0)
        else:
            world_score = 0.2
        for hand in frame["hands"]:
            label = hand["label"]
            trust = hand.get("joint_confidence") or []
            trust_score = float(np.clip(np.mean(trust), 0.0, 1.0)) if trust else float(hand.get("score") or 0.0)
            cv = frame_cv.get((label, frame["frame"]), 0.15)
            bone_score = float(np.clip(1.0 - cv / 0.30, 0.0, 1.0))
            raw = hand.get("raw_label")
            if hand.get("inferred"):
                identity_score = 0.3
            else:
                identity_score = 1.0 if not raw or raw == label else 0.0
            plausibility = 1.0
            state = hand.get("action")
            curl = float(hand.get("mean_curl") or 0.0)
            if label in prev_state and state:
                prev_action, prev_curl = prev_state[label]
                if (prev_action, state) in IMPLAUSIBLE_TRANSITIONS:
                    plausibility = 0.2
                elif abs(curl - prev_curl) > CURL_JUMP_RAD:
                    plausibility = 0.3
            if state:
                prev_state[label] = (state, curl)
            parts = {
                "trust": trust_score,
                "bone": bone_score,
                "identity": identity_score,
                "world": world_score,
                "plausibility": plausibility,
            }
            score = float(sum(WEIGHTS[name] * parts[name] for name in WEIGHTS))
            hand["quality"] = round(score, 3)
            hand["quality_parts"] = {name: round(value, 3) for name, value in parts.items()}
            for name in WEIGHTS:
                totals[name].append(parts[name])
            all_scores.append(score)

    overall = float(np.mean(all_scores)) if all_scores else None
    scored_segments = []
    for seg in segments:
        seg_scores = [
            hand["quality"]
            for frame in frames
            if seg["start_frame"] <= frame["frame"] <= seg["end_frame"]
            for hand in frame["hands"]
            if hand["label"] == seg["label"] and hand.get("quality") is not None
        ]
        if not seg_scores:
            continue
        mean = float(np.mean(seg_scores))
        scored_segments.append({**seg, "quality": round(mean, 3), "grade": grade_for(mean)})
    return {
        "overall": None if overall is None else round(overall, 3),
        "grade": None if overall is None else grade_for(overall),
        "components": {name: round(float(np.mean(values)), 3) for name, values in totals.items() if values},
        "weights": WEIGHTS,
        "segments": scored_segments,
        "notes": (
            "Composite of joint trust, bone stability, identity consistency, world support, "
            "and action plausibility. Heuristic, not ground-truth accuracy."
        ),
    }
