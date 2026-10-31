#!/usr/bin/env python3
"""Batch WiLoR 3D joints + mesh, written in the PS4 JSONL hand format.

Cloud GPU:
  bash scripts/setup_mesh_gpu.sh
  python scripts/run_mesh.py --input data/raw/clip.mp4 --merge outputs/clip.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.mesh import hands_from_predictions, load_wilor, merge_jsonl, save_mesh_npz, frame_record


def run_video(
    input_path: Path,
    output_dir: Path,
    *,
    batch_size: int = 8,
    stride: int = 1,
    max_frames: int | None = None,
    merge_path: Path | None = None,
    device: str | None = None,
    fp16: bool = True,
) -> dict:
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open {input_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pipe, faces = load_wilor(device=device, fp16=fp16)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = input_path.stem
    jsonl_path = output_dir / f"{stem}_mesh.jsonl"
    npz_path = output_dir / f"{stem}_mesh.npz"
    merged_path = output_dir / f"{stem}_merged.jsonl"

    chunks: list[dict] = []
    index = 0
    mesh_index = 0
    started = time.perf_counter()
    jsonl_path.write_text("", encoding="utf-8")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames is not None and index >= max_frames:
                break
            t_sec = index / fps
            if stride > 1 and index % stride != 0:
                index += 1
                continue
            outputs = pipe.predict(frame) or []
            hands, mesh_index = hands_from_predictions(outputs, width, height, mesh_index)
            for hand in hands:
                chunks.append(
                    {
                        "frame": index,
                        "label": hand["label"],
                        "world": hand["world"],
                        "vertices": hand["vertices"],
                        "cam_t": hand["cam_t"],
                    }
                )
            record = frame_record(index, t_sec, hands)
            with jsonl_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            index += 1
            if batch_size > 0 and index % batch_size == 0:
                print(f"mesh frames {index}", flush=True)
    finally:
        cap.release()

    save_mesh_npz(npz_path, chunks, faces)
    merged = None
    attached = 0
    if merge_path is not None:
        attached = merge_jsonl(merge_path, jsonl_path, merged_path)
        merged = merged_path
    elapsed = max(time.perf_counter() - started, 1e-6)
    return {
        "mesh_jsonl": str(jsonl_path),
        "mesh_npz": str(npz_path),
        "merged_jsonl": None if merged is None else str(merged),
        "frames_written": index,
        "hands": len(chunks),
        "mesh_hands_attached": attached,
        "fps": round(index / elapsed, 3),
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="WiLoR mesh batch into the PS4 hand format")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs")
    parser.add_argument("--merge", type=Path, default=None, help="Existing MediaPipe JSONL to attach mesh fields onto")
    parser.add_argument("--batch-size", type=int, default=8, help="Flush a progress line every N inferred frames")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda, cpu, or mps. Default: cuda if available")
    parser.add_argument("--fp32", action="store_true", help="Disable float16 (float16 is CUDA only)")
    args = parser.parse_args(argv)
    summary = run_video(
        args.input,
        args.output_dir,
        batch_size=args.batch_size,
        stride=max(1, args.stride),
        max_frames=args.max_frames,
        merge_path=args.merge,
        device=args.device,
        fp16=not args.fp32,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
