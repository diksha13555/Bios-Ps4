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
