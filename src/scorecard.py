"""Self-check without labelled ground truth."""

from __future__ import annotations

import numpy as np

from src.geometry import PARENT


def _bone_cv(frames: list[dict]) -> float | None:
    values: dict[tuple[str, int], list[float]] = {}
    for frame in frames:
        for hand in frame["hands"]:
            if hand.get("inferred"):
                continue
            points = np.asarray(hand["world"], dtype=np.float64)
            if points.shape[0] < 21:
                continue
            for child, parent in PARENT.items():
                length = float(np.linalg.norm(points[child] - points[parent]))
                if length > 1e-6:
                    values.setdefault((hand["label"], child), []).append(length)
    ratios = []
    for lengths in values.values():
        if len(lengths) < 3:
            continue
        mean = float(np.mean(lengths))
        if mean > 1e-6:
            ratios.append(float(np.std(lengths) / mean))
    if not ratios:
        return None
    return round(float(np.mean(ratios)), 4)


def _swaps(frames: list[dict]) -> int:
    count = 0
    for frame in frames:
        for hand in frame["hands"]:
            raw = hand.get("raw_label")
            if raw and raw != hand["label"] and not hand.get("inferred"):
                count += 1
    return count


def _umeyama(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Similarity alignment of src onto dst: scale, rotation, translation."""
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    xs, xd = src - mu_src, dst - mu_dst
    cov = (xd.T @ xs) / len(src)
    u, d, vt = np.linalg.svd(cov)
    s_flip = np.eye(3)
    if float(np.linalg.det(u @ vt)) < 0:
        s_flip[2, 2] = -1.0
    rot = u @ s_flip @ vt
    var = float((xs * xs).sum() / len(src))
    scale = float(np.trace(np.diag(d) @ s_flip) / var) if var > 1e-12 else 1.0
    return scale, rot, mu_dst - scale * (rot @ mu_src)


def _wilor_agreement(frames: list[dict]) -> dict:
    """MediaPipe vs WiLoR after rigid Procrustes alignment.

    Raw coordinates mix genuine disagreement with pose/scale conventions, so
    the fair comparison aligns each frame first (Umeyama similarity). The raw
    pre-alignment error is kept alongside for context.
    """
    raw_errors = []
    aligned_errors = []
    scales = []
    for frame in frames:
        for hand in frame["hands"]:
            mesh = hand.get("mesh")
            if not isinstance(mesh, dict) or not mesh.get("world"):
                continue
            world = np.asarray(hand["world"], dtype=np.float64)
            other = np.asarray(mesh["world"], dtype=np.float64)
            count = min(len(world), len(other), 21)
            if count < 21:
                continue
            world, other = world[:count], other[:count]
            raw_errors.append(float(np.linalg.norm(world - other, axis=1).mean()))
            scale, rot, tvec = _umeyama(other, world)
            aligned = scale * ((rot @ other.T).T) + tvec
            aligned_errors.append(float(np.linalg.norm(aligned - world, axis=1).mean()))
            scales.append(scale)
    if not aligned_errors:
        return {"frames": 0, "mean_joint_m": None}
    return {
        "frames": len(aligned_errors),
        "mean_joint_m": round(float(np.mean(aligned_errors)), 5),
        "mean_joint_raw_m": round(float(np.mean(raw_errors)), 5),
        "mean_scale": round(float(np.mean(scales)), 4),
        "note": "mean_joint_m is after Umeyama similarity alignment; mean_joint_raw_m is the pre-alignment error.",
    }


def build_scorecard(frames: list[dict], drift: dict) -> dict:
    return {
        "bone_length_cv": _bone_cv(frames),
        "identity_disagreements": _swaps(frames),
        "mediapipe_wilor": _wilor_agreement(frames),
        "world_drift": drift,
        "notes": {
            "bone_length_cv": "Mean coefficient of variation of kinematic bone lengths. Lower is more stable.",
            "identity_disagreements": "Frames where MediaPipe's raw Left/Right disagreed with the locked identity.",
            "mediapipe_wilor": "Mean joint distance after Procrustes alignment when a WiLoR mesh is attached. Null until a mesh merge exists.",
            "world_drift": "Wrist gap between a release and the next grasp. Translation is up to scale.",
        },
    }
