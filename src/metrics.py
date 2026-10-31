"""Tracking report for slides and honesty."""

from __future__ import annotations

import numpy as np


def _present(frame: dict, *, detected_only: bool) -> set[str]:
    out = set()
    for hand in frame["hands"]:
        if detected_only and hand.get("interpolated"):
            continue
        out.add(hand["label"])
    return out


def detection_share(frames: list[dict], *, detected_only: bool = False) -> dict[str, float]:
    if not frames:
        return {"left": 0.0, "right": 0.0, "either": 0.0, "both": 0.0}
    left = right = either = both = 0
    for frame in frames:
        ids = _present(frame, detected_only=detected_only)
        if "Left" in ids:
            left += 1
        if "Right" in ids:
            right += 1
        if ids:
            either += 1
        if "Left" in ids and "Right" in ids:
            both += 1
    n = float(len(frames))
    return {"left": left / n, "right": right / n, "either": either / n, "both": both / n}


def mean_score(frames: list[dict]) -> float:
    scores = [float(h["score"]) for frame in frames for h in frame["hands"] if not h.get("interpolated")]
    return float(np.mean(scores)) if scores else 0.0


def dropout_count(frames: list[dict]) -> int:
    count = 0
    prev = {"Left": False, "Right": False}
    for frame in frames:
        present = _present(frame, detected_only=True)
        for label in ("Left", "Right"):
            if prev[label] and label not in present:
                count += 1
            prev[label] = label in present
    return count


def mean_world_speed(frames: list[dict]) -> float:
    speeds: list[np.ndarray] = []
    for label in ("Left", "Right"):
        samples: list[tuple[float, np.ndarray]] = []
        for frame in frames:
            for hand in frame["hands"]:
                if hand["label"] == label:
                    samples.append((float(frame["t_sec"]), np.asarray(hand["world"], dtype=np.float64)))
                    break
        if len(samples) < 2:
            continue
        times = np.array([s[0] for s in samples])
        pts = np.stack([s[1] for s in samples])
        dt = np.clip(np.diff(times), 1e-6, None)
        dist = np.linalg.norm(np.diff(pts, axis=0), axis=2).mean(axis=1)
        speeds.append(dist / dt)
    if not speeds:
        return 0.0
    return float(np.mean(np.concatenate(speeds)))


def mean_reproj_error(frames: list[dict]) -> float | None:
    errs = [
        float(h["reproj_error_px"])
        for frame in frames
        for h in frame["hands"]
        if h.get("reproj_error_px") is not None and not h.get("interpolated")
    ]
    return float(np.mean(errs)) if errs else None


def build_accuracy_summary(
    raw: list[dict],
    filled: list[dict],
    *,
    reproj_px: float | None,
) -> dict:
    """Headline numbers for slides: quality when detected, coverage, stability."""
    conf = mean_score(raw)
    raw_share = detection_share(raw, detected_only=True)
    filled_share = detection_share(filled, detected_only=False)
    dropouts = dropout_count(raw)
    lead = (
        f"Mean detection confidence {conf:.0%} on real frames; "
        f"either-hand coverage {raw_share['either']:.0%} (raw) / {filled_share['either']:.0%} (with hold+fill); "
        f"{dropouts} dropout events."
    )
    if reproj_px is not None:
        lead += f" Mean PnP reprojection {reproj_px:.1f} px."
    return {
        "mean_detection_confidence": round(conf, 4),
        "coverage_raw": {k: round(v, 4) for k, v in raw_share.items()},
        "coverage_with_hold_and_fill": {k: round(v, 4) for k, v in filled_share.items()},
        "dropout_events": dropouts,
        "mean_pnp_reproj_px": round(reproj_px, 3) if reproj_px is not None else None,
        "pitch_lead": lead,
    }


def build_report(raw: list[dict], filled: list[dict], *, fps: float, fps_hands: float, config: dict) -> dict:
    raw_speed = mean_world_speed(filled)
    reproj = mean_reproj_error(filled)
    return {
        "frames": len(filled),
        "fps": fps,
        "fps_hands": round(fps_hands, 3),
        "detection_share": detection_share(filled, detected_only=False),
        "detection_share_detected_only": detection_share(raw, detected_only=True),
        "mean_score": mean_score(raw),
        "dropout_count": dropout_count(raw),
        "world_landmark_speed_mps": raw_speed,
        "accuracy": build_accuracy_summary(raw, filled, reproj_px=reproj),
        "config": config,
        "notes": {
            "world": "MediaPipe world_landmarks are hand-centric meters, not a factory frame.",
            "speed": "Mean speed of 21 world points; lower usually means less jitter.",
            "accuracy": "Lead with mean_detection_confidence; explain coverage vs egocentric visibility.",
        },
    }
