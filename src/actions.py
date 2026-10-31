"""Grasp, release, reach, hold, and idle from wrist motion and finger curl.

Grasp names follow the coarse power / precision / lateral split of Feix et al.'s
GRASP taxonomy. They are labels for a segment, not a claim of the full 33-class set.
"""

from __future__ import annotations

import numpy as np

from src.export_sim import finger_curls

ACTIONS = ("idle", "reach", "grasp", "hold", "release")


def _spread(points: np.ndarray) -> float:
    tips = points[[4, 8, 12, 16, 20]]
    return float(np.mean(np.linalg.norm(tips - tips.mean(axis=0), axis=1)))


def _grasp_name(curls: dict[str, float]) -> str | None:
    fingers = [curls["index"], curls["middle"], curls["ring"], curls["pinky"]]
    if min(fingers) > 1.0 and curls["thumb"] > 0.6:
        return "power"
    if curls["thumb"] > 0.8 and curls["index"] > 0.8 and curls["ring"] < 0.7 and curls["pinky"] < 0.7:
        return "precision"
    if curls["thumb"] > 0.8 and curls["index"] < 0.9 and max(fingers) < 1.0:
        return "lateral"
    if max(fingers) > 0.9:
        return "other"
    return None


def _position(hand: dict) -> np.ndarray:
    points = np.asarray(hand.get("world_fixed") or hand["world"], dtype=np.float64)
    return points[0, :3]


def annotate_actions(frames: list[dict], fps: float) -> list[dict]:
    default_dt = 1.0 / max(float(fps), 1e-6)
    labels = {hand["label"] for frame in frames for hand in frame["hands"]}
    segments: list[dict] = []
    for label in sorted(labels):
        series = []
        for frame in frames:
            for hand in frame["hands"]:
                if hand["label"] == label:
                    series.append((frame, hand))
        previous_curl = None
        previous_frame_num = None
        previous_t = None
        states = []
        for frame, hand in series:
            points = np.asarray(hand.get("world") , dtype=np.float64)
            curls = finger_curls(points) if points.shape[0] >= 21 else {}
            mean_curl = float(np.mean(list(curls.values()))) if curls else 0.0
            dt = default_dt
            gap_break = False
            if previous_t is not None:
                dt = frame["t_sec"] - previous_t
                if dt <= 1e-6:
                    dt = default_dt
                gap_break = frame["frame"] - previous_frame_num > 1
            speed = 0.0
            if states:
                prev_hand = series[len(states) - 1][1]
                speed = float(np.linalg.norm(_position(hand) - _position(prev_hand)) / dt)
            if hand.get("inferred") or gap_break:
                # Filled or reappearing hands carry the prior state; grasp/release
                # events only fire on continuously tracked, real measurements.
                state = states[-1] if states else "idle"
                curl_delta = 0.0
                previous_curl = None
            else:
                curl_delta = 0.0 if previous_curl is None else (mean_curl - previous_curl) / dt
                previous_curl = mean_curl
                if curl_delta > 1.5 and mean_curl > 0.5:
                    state = "grasp"
                elif curl_delta < -1.5 and mean_curl > 0.35:
                    state = "release"
                elif mean_curl >= 1.1 and speed < 0.04:
                    state = "hold"
                elif speed >= 0.04 and mean_curl < 1.0:
                    state = "reach"
                else:
                    state = "idle"
            grasp = _grasp_name(curls) if state in {"grasp", "hold"} else None
            hand["action"] = state
            hand["grasp"] = grasp
            hand["mean_curl"] = round(mean_curl, 4)
            states.append(state)
            previous_frame_num = frame["frame"]
            previous_t = frame["t_sec"]
        if not series:
            continue
        start = 0
        for index in range(1, len(states) + 1):
            if index == len(states) or states[index] != states[start]:
                frame, hand = series[start]
                end_frame, _ = series[index - 1]
                segments.append(
                    {
                        "label": label,
                        "action": states[start],
                        "grasp": hand.get("grasp"),
                        "start_frame": frame["frame"],
                        "end_frame": end_frame["frame"],
                        "start_sec": frame["t_sec"],
                        "end_sec": end_frame["t_sec"],
                    }
                )
                start = index
    return segments
