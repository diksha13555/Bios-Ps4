#!/usr/bin/env python3
"""Evaluate pipeline JSONL against hand-labelled ground truth."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.eval_gt import evaluate_ground_truth, load_jsonl, merge_operational_report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Ground-truth hand tracking evaluation")
    p.add_argument("--predictions", type=Path, required=True, help="Pipeline output JSONL")
    p.add_argument("--labels", type=Path, required=True, help="Ground-truth annotation JSONL")
    p.add_argument("--report", type=Path, default=None, help="Optional JSON report path")
    p.add_argument("--operational-report", type=Path, default=None, help="Optional pipeline _report.json")
    p.add_argument("--match-dist", type=float, default=0.08, help="Wrist match threshold (normalized xy)")
    p.add_argument("--pck-threshold", type=float, default=0.05, help="PCK threshold (normalized xy)")
    args = p.parse_args(argv)

    if not args.predictions.exists():
        print(f"Predictions not found: {args.predictions}", file=sys.stderr)
        return 2
    if not args.labels.exists():
        print(f"Labels not found: {args.labels}", file=sys.stderr)
        print(
            "Create annotation JSONL with one object per labelled frame. "
            "See README.md section 'Ground-truth annotation format'.",
            file=sys.stderr,
        )
        return 2

    preds = load_jsonl(args.predictions)
    labels = load_jsonl(args.labels)
    body = evaluate_ground_truth(
        preds,
        labels,
        match_dist=args.match_dist,
        pck_threshold=args.pck_threshold,
    )
    if args.operational_report and args.operational_report.exists():
        op = json.loads(args.operational_report.read_text(encoding="utf-8"))
        body = {**body, **merge_operational_report(op)}

    text = json.dumps(body, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    if body.get("status") != "ok":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
