from src.smooth import interpolate_gaps


def _hand(label, x):
    pt = [x, 0.5, 0.0]
    return {"label": label, "score": 0.9, "world": [pt] * 21, "image": [pt] * 21, "tracked": True, "interpolated": False}


def test_gap_fill():
    frames = [
        {"frame": 0, "t_sec": 0.0, "hands": [_hand("Left", 0.3)]},
        {"frame": 1, "t_sec": 0.033, "hands": []},
        {"frame": 2, "t_sec": 0.066, "hands": [_hand("Left", 0.35)]},
    ]
    interpolate_gaps(frames, max_gap=5)
    assert frames[1]["hands"][0]["interpolated"] is True
