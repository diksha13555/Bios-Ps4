#!/usr/bin/env python3
"""PS4: 3D hand motion beside the source video."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.hands import DEFAULT_MODEL, download_model
from src.pipeline import run_video


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="PS4 hand motion pipeline")
    p.add_argument("--input", type=Path, required=False)
    p.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--download-model", action="store_true")
    p.add_argument("--max-side", type=int, default=1280)
    p.add_argument("--viz-max-side", type=int, default=960)
    p.add_argument("--no-clahe", action="store_true")
    p.add_argument("--no-camera-pose", action="store_true")
    p.add_argument("--no-world", action="store_true", help="Disable fixed-world trajectories and world sim export")
    p.add_argument("--world-smooth-window", type=int, default=11, help="Savgol window (frames) for world wrist trajectories")
    p.add_argument("--world-zoom", type=float, default=1.25, help="Display-only 3D zoom, clamped to 1.0-2.0")
    p.add_argument(
        "--no-ground",
        action="store_true",
        help="Disable contact anchoring and hand-anchored scale recovery (keeps raw up-to-scale world)",
    )
    p.add_argument(
        "--high-accuracy",
        action="store_true",
        help="Use 1920px+ detector input, wider ROI recovery, and recall-oriented thresholds (slower)",
    )
    p.add_argument("--max-frames", type=int, default=None)
    p.add_argument("--max-gap", type=int, default=5)
    p.add_argument("--min-detection-confidence", type=float, default=0.5)
    p.add_argument("--min-presence-confidence", type=float, default=0.5)
    p.add_argument("--min-tracking-confidence", type=float, default=0.5)
    args = p.parse_args(argv)
    if args.download_model:
        print(download_model(args.model))
        return 0
    if args.input is None:
        print("Usage: python scripts/run_ps4.py --input data/raw/clip.mp4", file=sys.stderr)
        return 2
    paths = run_video(
        args.input,
        args.output_dir,
        model_path=args.model,
        max_side=args.max_side,
        viz_max_side=args.viz_max_side,
        clahe=not args.no_clahe,
        max_frames=args.max_frames,
        max_gap=args.max_gap,
        camera_pose=not args.no_camera_pose,
        world=not args.no_world,
        world_smooth_window=args.world_smooth_window,
        world_zoom=args.world_zoom,
        ground=not args.no_ground,
        high_accuracy=args.high_accuracy,
        min_detection_confidence=args.min_detection_confidence,
        min_presence_confidence=args.min_presence_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
    )
    body = paths["report_body"]
    print(
        json.dumps(
            {
                "side_by_side": str(paths["side_by_side"]),
                "jsonl": str(paths["jsonl"]),
                "report": str(paths["report"]),
                "sim": str(paths["sim"]),
                "world_sim": None if paths["world_sim"] is None else str(paths["world_sim"]),
                "world_csv": None if paths["world_csv"] is None else str(paths["world_csv"]),
                "world_valid_share": body.get("world", {}).get("valid_share"),
                "touchtrace": {
                    "overall": body.get("touchtrace", {}).get("overall"),
                    "grade": body.get("touchtrace", {}).get("grade"),
                },
                "grounding_applied": body.get("grounding", {}).get("applied"),
                "detection_share": body["detection_share"],
                "accuracy": body["accuracy"],
                "fps_end_to_end": body["fps_end_to_end"],
                "fps_hands": body["fps_hands"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
