#!/usr/bin/env python3
"""Click-based ground-truth annotation for the TouchTrace evaluator.

Annotate real frames with mouse clicks. Two modes:
  --points 21  full MediaPipe landmark order (best PCK quality)
  --points 6   wrist + five fingertips (fast; PCK on 6 joints)

Keys:
  click     place the next landmark
  L / R     current hand is Left / Right (default Right)
  Enter     save the current hand and move on
  U         undo last point
  N         save and jump to the next frame
  P         previous frame
  S         mark frame as no-hand (negative example)
  Q / Esc   quit (progress is already saved)

Output JSONL matches the evaluator: full GT uses 21 points; partial GT adds
`indices` so PCK is computed on the annotated joints only. Labels are human
clicks only, never model output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

JOINT_NAMES = {
    0: "wrist", 1: "thumb CMC", 2: "thumb MCP", 3: "thumb IP", 4: "thumb tip",
    5: "index MCP", 6: "index PIP", 7: "index DIP", 8: "index tip",
    9: "middle MCP", 10: "middle PIP", 11: "middle DIP", 12: "middle tip",
    13: "ring MCP", 14: "ring PIP", 15: "ring DIP", 16: "ring tip",
    17: "pinky MCP", 18: "pinky PIP", 19: "pinky DIP", 20: "pinky tip",
}
SIX_POINT = [0, 4, 8, 12, 16, 20]


class Annotator:
    def __init__(self, video: Path, output: Path, start: int, stride: int, max_frames: int, points: int):
        self.cap = cv2.VideoCapture(str(video))
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open {video}")
        self.output = output
        self.output.parent.mkdir(parents=True, exist_ok=True)
        self.indices = list(range(21)) if points == 21 else SIX_POINT
        self.frames = list(range(start, int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT)), stride))[:max_frames]
        self.position = 0
        self.label = "Right"
        self.clicks: list[tuple[int, int]] = []
        self.saved: dict[int, list[dict]] = {}
        self.scale = 1.0
        self.window = "TouchTrace annotation"
        cv2.namedWindow(self.window, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.window, self.on_click)

    def on_click(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(self.clicks) < len(self.indices):
            self.clicks.append((x, y))

    def current_frame(self):
        index = self.frames[self.position]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = self.cap.read()
        if not ok:
            return None, None
        height, width = frame.shape[:2]
        self.scale = min(1.0, 1100.0 / width)
        view = frame if self.scale == 1.0 else cv2.resize(frame, (int(width * self.scale), int(height * self.scale)))
        return index, (frame, view)

    def save_hand(self, index: int, original_shape):
        if not self.clicks:
            return
        height, width = original_shape[:2]
        points = [
            [round(x / self.scale / width, 5), round(y / self.scale / height, 5), 0.0]
            for x, y in self.clicks
        ]
        used = self.indices[: len(points)]
        hand = {"label": self.label, "image": points}
        if len(used) != 21:
            hand["indices"] = used
        self.saved.setdefault(index, []).append(hand)
        self.clicks = []

    def flush(self):
        with self.output.open("w", encoding="utf-8") as handle:
            for index in sorted(self.saved):
                handle.write(json.dumps({"frame": index, "hands": self.saved[index]}) + "\n")

    def run(self) -> int:
        print(f"Annotating {len(self.frames)} frames -> {self.output}")
        while 0 <= self.position < len(self.frames):
            index, pair = self.current_frame()
            if pair is None:
                break
            original, view = pair
            view = view.copy()
            next_joint = self.indices[len(self.clicks)] if len(self.clicks) < len(self.indices) else None
            status = (
                f"frame {index} ({self.position + 1}/{len(self.frames)}) | {self.label} | "
                + (f"click {next_joint}: {JOINT_NAMES[next_joint]}" if next_joint is not None else "Enter=save hand N=next frame")
            )
            cv2.putText(view, status, (12, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            for number, (x, y) in enumerate(self.clicks):
                cv2.circle(view, (x, y), 5, (0, 220, 255), -1)
                cv2.putText(view, str(self.indices[number]), (x + 7, y - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 255), 1)
            cv2.imshow(self.window, view)
            key = cv2.waitKey(0) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("u") and self.clicks:
                self.clicks.pop()
            elif key == ord("l"):
                self.label = "Left"
            elif key == ord("r"):
                self.label = "Right"
            elif key == ord("s"):
                self.saved[index] = []
                self.clicks = []
                self.position += 1
            elif key in (13, 10):
                self.save_hand(index, original.shape)
                self.flush()
            elif key == ord("n"):
                self.save_hand(index, original.shape)
                self.flush()
                self.position += 1
            elif key == ord("p"):
                self.position = max(0, self.position - 1)
                self.clicks = []
        self.flush()
        self.cap.release()
        cv2.destroyAllWindows()
        print(f"Saved {len(self.saved)} labelled frames to {self.output}")
        return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--max-frames", type=int, default=250)
    parser.add_argument("--points", type=int, choices=(21, 6), default=21)
    args = parser.parse_args(argv)
    return Annotator(args.video, args.output, args.start, args.stride, args.max_frames, args.points).run()


if __name__ == "__main__":
    raise SystemExit(main())
