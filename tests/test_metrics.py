from src.metrics import build_accuracy_summary, detection_share, mean_reproj_error


def _frame(hands):
    return {"t_sec": 0.0, "hands": hands}


def test_accuracy_summary_pitch_lead():
    raw = [
        _frame([{"label": "Left", "score": 0.9, "interpolated": False, "world": [[0, 0, 0]] * 21}]),
        _frame([]),
    ]
    filled = [
        _frame([{"label": "Left", "score": 0.9, "interpolated": False, "world": [[0, 0, 0]] * 21}]),
        _frame([{"label": "Left", "score": 0.9, "interpolated": True, "world": [[0, 0, 0]] * 21}]),
    ]
    acc = build_accuracy_summary(raw, filled, reproj_px=4.2)
    assert acc["mean_detection_confidence"] == 0.9
    assert acc["coverage_raw"]["left"] == 0.5
    assert acc["coverage_with_hold_and_fill"]["left"] == 1.0
    assert "90%" in acc["pitch_lead"]
    assert acc["mean_pnp_reproj_px"] == 4.2


def test_mean_reproj_error_skips_interpolated():
    frames = [
        _frame(
            [
                {"label": "Left", "interpolated": False, "reproj_error_px": 10.0},
                {"label": "Right", "interpolated": True, "reproj_error_px": 999.0},
            ]
        )
    ]
    assert mean_reproj_error(frames) == 10.0


def test_detection_share_either():
    frames = [{"hands": [{"label": "Right", "interpolated": False}]}]
    assert detection_share(frames, detected_only=True)["either"] == 1.0
