"""Optional solvePnP into an approximate camera frame."""

from __future__ import annotations

import cv2
import numpy as np

_SOLVERS = (cv2.SOLVEPNP_SQPNP, cv2.SOLVEPNP_ITERATIVE, cv2.SOLVEPNP_EPNP)


def intrinsics(width: int, height: int) -> np.ndarray:
    focal = float(max(width, height))
    return np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)


def hand_to_camera(world, image_norm, matrix, width, height, max_err: float = 120.0) -> dict | None:
    obj = np.asarray(world, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(image_norm, dtype=np.float64).reshape(-1, 3)
    if obj.shape[0] < 6:
        return None
    px = np.column_stack([img[:, 0] * width, img[:, 1] * height])
    dist = np.zeros(4)
    best = None
    for flag in _SOLVERS:
        try:
            ok, rvec, tvec = cv2.solvePnP(obj, px, matrix, dist, flags=flag)
        except cv2.error:
            continue
        if not ok:
            continue
        rot, _ = cv2.Rodrigues(rvec)
        cam = (rot @ obj.T).T + tvec.reshape(1, 3)
        if float(np.mean(cam[:, 2])) <= 0:
            continue
        proj, _ = cv2.projectPoints(obj, rvec, tvec, matrix, dist)
        per_joint = np.linalg.norm(proj.reshape(-1, 2) - px, axis=1)
        err = float(per_joint.mean())
        if best is None or err < best[0]:
            best = (err, cam.tolist(), rvec.reshape(-1).tolist(), tvec.reshape(-1).tolist(), per_joint.tolist())
    if best is None or best[0] > max_err:
        return None
    return {
        "camera": best[1],
        "rvec": best[2],
        "tvec": best[3],
        "reproj_error_px": best[0],
        "joint_reproj_px": best[4],
    }


def attach_camera(frames: list[dict], width: int, height: int) -> None:
    k = intrinsics(width, height)
    for frame in frames:
        for hand in frame["hands"]:
            pose = hand_to_camera(hand["world"], hand["image"], k, width, height)
            if pose:
                hand.update(pose)
            else:
                hand["camera"] = None
                hand["reproj_error_px"] = None
