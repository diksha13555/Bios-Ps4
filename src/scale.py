"""Ego-scale recovery from hand anthropometrics.

MediaPipe's world landmarks are nominal hand units, and the monocular
camera-motion fallback leaves trajectory translation unscaled. The operator's
own hand is a calibration object present in every frame: trusted bone lengths
pin down a per-clip size scale, and stationary grasps/holds pin down the
trajectory translation scale. Both estimates are heuristics, not ground truth.
"""

from __future__ import annotations

import numpy as np

from src.geometry import PARENT

# Adult anthropometric means in meters, keyed by MediaPipe child joint index.
REFERENCE_BONE_M = {
    5: 0.068,   # wrist -> index MCP (2nd metacarpal)
    9: 0.072,   # wrist -> middle MCP (3rd metacarpal)
    13: 0.066,  # wrist -> ring MCP (4th metacarpal)
    17: 0.055,  # wrist -> pinky MCP (5th metacarpal)
    6: 0.040,   # index proximal phalanx
    10: 0.045,  # middle proximal phalanx
    14: 0.041,  # ring proximal phalanx
    18: 0.032,  # pinky proximal phalanx
    7: 0.025,   # index intermediate phalanx
    11: 0.028,  # middle intermediate phalanx
    2: 0.032,   # thumb CMC -> MCP
    3: 0.030,   # thumb MCP -> IP
}
TRUST_MIN = 0.7
MIN_BONES = 4
SCALE_CLIP = (0.5, 2.0)
VO_SCALE_CLIP = (0.05, 50.0)
CONTACT_ACTIONS = frozenset({"hold", "grasp"})


def estimate_hand_scale(frames: list[dict]) -> dict:
    """Ratio of anthropometric reference to observed bone length, per trusted bone.

    scale=1 means the clip's nominal hand units already look metric.
    """
    samples: dict[tuple[str, int], list[float]] = {}
    for frame in frames:
        for hand in frame["hands"]:
            if hand.get("inferred"):
                continue
            trust = hand.get("joint_confidence") or []
            points = np.asarray(hand.get("world") or [], dtype=np.float64)
            if points.ndim != 2 or points.shape[0] < 21:
                continue
            for child, parent in PARENT.items():
                if child not in REFERENCE_BONE_M:
                    continue
                if child < len(trust) and float(trust[child]) < TRUST_MIN:
                    continue
                length = float(np.linalg.norm(points[child] - points[parent]))
                if length > 1e-6:
                    samples.setdefault((hand["label"], child), []).append(length)
    per_label: dict[str, float] = {}
    all_ratios: list[float] = []
    for label in {key[0] for key in samples}:
        ratios = []
        for (bone_label, child), values in samples.items():
            if bone_label != label or len(values) < 3:
                continue
            observed = float(np.median(values))
            if observed > 1e-6:
                ratios.append(REFERENCE_BONE_M[child] / observed)
        if len(ratios) >= MIN_BONES:
            per_label[label] = round(float(np.clip(np.median(ratios), *SCALE_CLIP)), 4)
            all_ratios.extend(ratios)
    if len(all_ratios) < MIN_BONES:
        return {"scale": None, "per_label": per_label, "bones_used": 0}
    return {
        "scale": round(float(np.clip(np.median(all_ratios), *SCALE_CLIP)), 4),
        "per_label": per_label,
        "bones_used": len(all_ratios),
    }


def estimate_vo_scale(frames: list[dict], *, actions: frozenset = CONTACT_ACTIONS) -> dict:
    """Scale for the camera-motion translation, from stationary-contact frames.

    During a grasp/hold the hand is stationary in the world, so the true camera
    translation step equals the hand's apparent step in the camera frame
    (nominal hand units). The median ratio of that step's norm to the VO step's
    norm converts the whole camera trajectory into the same units.
    """
    ratios: list[float] = []
    prev: dict[str, dict] = {}
    for frame in frames:
        pose = frame.get("camera_pose_world") or {}
        translation = pose.get("translation")
        if translation is None or pose.get("rotation") is None or not pose.get("valid"):
            prev = {}
            continue
        t_curr = np.asarray(translation, dtype=np.float64).reshape(3)
        for hand in frame["hands"]:
            if hand.get("action") not in actions or hand.get("inferred"):
                continue
            camera = hand.get("camera")
            if camera is None:
                continue
            wrist = np.asarray(camera, dtype=np.float64)[0, :3]
            last = prev.get(hand["label"])
            if last is not None and frame["frame"] == last["frame"] + 1 and last["action"] in actions:
                n_cam = float(np.linalg.norm(wrist - last["wrist"]))
                n_vo = float(np.linalg.norm(t_curr - last["t"]))
                if n_cam > 1e-5 and n_vo > 1e-6:
                    ratios.append(n_cam / n_vo)
            prev[hand["label"]] = {
                "frame": frame["frame"],
                "wrist": wrist,
                "t": t_curr,
                "action": hand.get("action"),
            }
    if len(ratios) < 5:
        return {"vo_scale": None, "pairs": len(ratios)}
    return {"vo_scale": round(float(np.clip(np.median(ratios), *VO_SCALE_CLIP)), 4), "pairs": len(ratios)}
