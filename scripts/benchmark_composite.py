#!/usr/bin/env python3
"""Run the composite detection benchmark end to end.

Real hand pixels, exact compositing ground truth. Reports detection
precision/recall, wrist localization error, and identity switches.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.composite import generate_composite, wrist_localization
from src.eval_gt import evaluate_ground_truth, load_jsonl
from src.pipeline import run_video


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-video", type=Path, default=ROOT / "data/raw/MTL-SGF00106-021.mp4")
    parser.add_argument("--source-predictions", type=Path, default=ROOT / "outputs/MTL-SGF00106-021.jsonl")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/benchmark")
    parser.add_argument("--frames", type=int, default=240)
    args = parser.parse_args(argv)
    generated = generate_composite(args.source_video, args.source_predictions, args.output_dir, n_frames=args.frames)
    # CLAHE's local equalization destroys the feathered composite's contrast;
    # the composite is well lit, so run without it and record that choice.
    paths = run_video(generated["video"], args.output_dir, clahe=False)
    evaluation = evaluate_ground_truth(
        load_jsonl(Path(paths["jsonl"])),
        load_jsonl(generated["labels"]),
    )
    evaluation["wrist_localization"] = wrist_localization(Path(paths["jsonl"]), generated["labels"])
    evaluation["benchmark"] = {
        "type": "composite (real hand crop, synthetic background)",
        "ground_truth": "wrist + bbox from the compositing transform; no model-output labels",
        "scope": "Detection, wrist localization, identity stability. Landmark PCK requires manual annotation.",
        "source_video": str(args.source_video),
        "frames": args.frames,
    }
    report_path = args.output_dir / "composite_eval.json"
    report_path.write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    print(json.dumps(evaluation, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
