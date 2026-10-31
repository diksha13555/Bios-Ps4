"""PS4 video pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path

import cv2

from src.camera_frame import attach_camera
from src.export_sim import export_sim
from src.hands import DEFAULT_MODEL, HandRuntime, to_public_hand
from src.metrics import build_report
from src.smooth import interpolate_gaps
from src.visualize import SkeletonPanel, compose, resize_max_side


def run_video(
    input_path: Path,
    output_dir: Path,
    *,
    model_path: Path = DEFAULT_MODEL,
    max_side: int = 1280,
    viz_max_side: int = 960,
    clahe: bool = True,
    max_frames: int | None = None,
    max_gap: int = 5,
    camera_pose: bool = True,
    min_detection_confidence: float = 0.5,
    min_presence_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> dict:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {input_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    runtime = HandRuntime(
        model_path,
        max_side=max_side,
        clahe=clahe,
        min_detection_confidence=min_detection_confidence,
        min_presence_confidence=min_presence_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    side_path = output_dir / f"{stem}_sidebyside.mp4"
    jsonl_path = output_dir / f"{stem}.jsonl"
    report_path = output_dir / f"{stem}_report.json"
    sim_path = output_dir / f"{stem}_sim.json"

    raw_frames: list[dict] = []
    hand_time = 0.0
    started = time.perf_counter()
    panel = None
    writer = None
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and index >= max_frames:
                break
            t_sec = index / fps
            hands, dt = runtime.step(frame, t_sec)
            hand_time += dt
            public = [to_public_hand(h) for h in hands if not h.get("interpolated")]
            raw_frames.append({"frame": index, "t_sec": round(t_sec, 4), "hands": public})
            filled_hands = [to_public_hand(h) for h in hands]
            view = resize_max_side(frame, viz_max_side)
            if panel is None:
                panel = SkeletonPanel(view.shape[1], view.shape[0])
            elapsed = max(time.perf_counter() - started, 1e-6)
            share = _share_quick(raw_frames)
            hud = f"FPS {index / elapsed:.1f}  hands {share['either']:.0%}  L {share['left']:.0%}  R {share['right']:.0%}"
            out = compose(view, filled_hands, panel, hud)
            if writer is None:
                writer = cv2.VideoWriter(
                    str(side_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out.shape[1], out.shape[0])
                )
            writer.write(out)
            index += 1
    finally:
        cap.release()
        runtime.close()
        if panel:
            panel.close()
        if writer:
            writer.release()

    if not raw_frames:
        raise RuntimeError("no frames")

    processed = interpolate_gaps([dict(f, hands=[dict(h) for h in f["hands"]]) for f in raw_frames], max_gap=max_gap)
    if camera_pose:
        attach_camera(processed, width, height)

    with jsonl_path.open("w", encoding="utf-8") as f:
        for frame in processed:
            f.write(json.dumps(_json_frame(frame)) + "\n")

    total = max(time.perf_counter() - started, 1e-6)
    report = build_report(
        raw_frames,
        processed,
        fps=fps,
        fps_hands=index / max(hand_time, 1e-6),
        config={
            "max_side": max_side,
            "clahe": clahe,
            "max_gap": max_gap,
            "min_detection_confidence": min_detection_confidence,
            "min_presence_confidence": min_presence_confidence,
            "min_tracking_confidence": min_tracking_confidence,
        },
    )
    report["fps_end_to_end"] = round(index / total, 3)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    sim_path.write_text(json.dumps(export_sim(processed, fps=fps, source=input_path.name), indent=2), encoding="utf-8")
    return {"side_by_side": side_path, "jsonl": jsonl_path, "report": report_path, "sim": sim_path, "report_body": report}


def _share_quick(frames):
    from src.metrics import detection_share

    return detection_share(frames, detected_only=True)


def _json_frame(frame: dict) -> dict:
    hands = []
    for h in frame["hands"]:
        item = {
            "label": h["label"],
            "score": round(float(h["score"]), 4),
            "tracked": h.get("tracked", True),
            "interpolated": h.get("interpolated", False),
            "world": _round3(h["world"]),
            "image": _round3(h["image"]),
        }
        if h.get("camera") is not None:
            item["camera"] = _round3(h["camera"])
            item["reproj_error_px"] = h.get("reproj_error_px")
        hands.append(item)
    return {"frame": frame["frame"], "t_sec": frame["t_sec"], "hands": hands}


def _round3(points):
    import numpy as np

    return np.round(np.asarray(points), 6).tolist()
