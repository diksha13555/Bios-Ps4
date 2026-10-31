"""Contact-driven physics grounding.

Two ideas combined. First, the operator's hand calibrates the scene: hand bone
lengths give a size scale, and stationary contacts (grasp/hold) give the camera
translation scale, because a hand resting on an object is not moving in the
world. Second, each contact is a physics constraint: while the hand holds an
object, its wrist cannot drift. Anchored contacts remove exactly that drift.

All outputs are approximate: nominal hand units scaled toward metric by
heuristics. Raw world_fixed fields are never mutated; grounded values are
written to world_grounded alongside them.
"""

from __future__ import annotations

import numpy as np

from src.scale import CONTACT_ACTIONS, estimate_hand_scale, estimate_vo_scale

BLEND_FRAMES = 5


def _anchor_runs(flags: list[bool], min_len: int, *, gap_tolerance: int = 1) -> list[tuple[int, int]]:
    """Contact runs of True flags, tolerating one-frame label flickers."""
    closed = list(flags)
    for i, flag in enumerate(flags):
        if flag:
            continue
        left = i - 1 >= 0 and flags[i - 1]
        right = i + 1 < len(flags) and flags[i + 1]
        if gap_tolerance and left and right:
            closed[i] = True
    runs: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(closed):
        if flag:
            if start is None:
                start = i
        else:
            if start is not None and i - start >= min_len:
                runs.append((start, i - 1))
            start = None
    if start is not None and len(closed) - start >= min_len:
        runs.append((start, len(closed) - 1))
    return runs


def ground_contacts(frames: list[dict], *, fps: float, min_anchor_sec: float = 0.1) -> dict:
    """Attach world_grounded joints (approximate meters) using contact anchors.

    Returns a summary with applied=False when either scale estimate is
    unavailable, in which case frames are left untouched.
    """
    hand_scale = estimate_hand_scale(frames)
    vo_scale = estimate_vo_scale(frames)
    hs, vs = hand_scale["scale"], vo_scale["vo_scale"]
    result = {
        "applied": False,
        "hand_scale": hand_scale,
        "vo_scale": vs,
        "vo_pairs": vo_scale["pairs"],
        "anchors": [],
        "wrists": {},
        "units": "approximate meters (hand-anchored)",
    }
    if hs is None or vs is None:
        return result
    min_len = max(2, int(round(min_anchor_sec * fps)))
    labels = sorted({hand["label"] for frame in frames for hand in frame["hands"]})
    for label in labels:
        series = []
        for index, frame in enumerate(frames):
            pose = frame.get("camera_pose_world") or {}
            if not pose.get("valid") or pose.get("rotation") is None or pose.get("translation") is None:
                continue
            hand = next((h for h in frame["hands"] if h["label"] == label and h.get("camera") is not None), None)
            if hand is None:
                continue
            rot = np.asarray(pose["rotation"], dtype=np.float64)
            tvec = np.asarray(pose["translation"], dtype=np.float64).reshape(3)
            camera = np.asarray(hand["camera"], dtype=np.float64)
            if camera.ndim != 2 or camera.shape[0] < 21:
                continue
            points = hs * ((rot @ camera.T).T + vs * tvec)
            series.append(
                {
                    "hand": hand,
                    "frame": frame["frame"],
                    "t_sec": frame["t_sec"],
                    "points": points,
                    "anchor": hand.get("action") in CONTACT_ACTIONS and not hand.get("inferred"),
                }
            )
        if len(series) < 2:
            continue
        wrists = np.asarray([sample["points"][0, :3] for sample in series])
        corrected = wrists.copy()
        flags = [sample["anchor"] for sample in series]
        for start, end in _anchor_runs(flags, min_len):
            anchor = np.median(corrected[start : end + 1], axis=0)
            drift_removed = float(np.linalg.norm(corrected[end] - anchor))
            blend_in = min(BLEND_FRAMES, start)
            blend_out = min(BLEND_FRAMES, len(series) - 1 - end)
            for i in range(start - blend_in, start):
                alpha = (i - (start - blend_in) + 1) / (blend_in + 1)
                corrected[i] = (1 - alpha) * corrected[i] + alpha * anchor
            for i in range(end + 1, end + 1 + blend_out):
                alpha = (end + 1 + blend_out - i) / (blend_out + 1)
                corrected[i] = (1 - alpha) * corrected[i] + alpha * anchor
            corrected[start : end + 1] = anchor
            result["anchors"].append(
                {
                    "label": label,
                    "start_frame": series[start]["frame"],
                    "end_frame": series[end]["frame"],
                    "start_sec": series[start]["t_sec"],
                    "end_sec": series[end]["t_sec"],
                    "anchor_xyz": [round(float(v), 5) for v in anchor],
                    "drift_removed_m": round(drift_removed, 5),
                }
            )
        for i, sample in enumerate(series):
            offset = corrected[i] - wrists[i]
            sample["hand"]["world_grounded"] = (sample["points"] + offset).tolist()
            result["wrists"][(sample["frame"], label)] = [round(float(v), 5) for v in corrected[i]]
    result["applied"] = bool(result["anchors"])
    if not result["applied"]:
        for frame in frames:
            for hand in frame["hands"]:
                hand.pop("world_grounded", None)
        result["wrists"] = {}
    return result
