from src.hands import assign_tracks, landmarks_bbox
from src.metrics import detection_share


def _det(label, box):
    c = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2, 0.0]
    img = [list(c) for _ in range(21)]
    return {"label": label, "score": 0.9, "image": img, "world": img, "bbox": box}


def test_side_swap_keeps_identity():
    box = (0.2, 0.2, 0.5, 0.6)
    tracks, out = assign_tracks([], [_det("Left", box)])
    tracks, out = assign_tracks(tracks, [_det("Right", (0.22, 0.22, 0.52, 0.62))])
    assert out[0]["identity"] == "Left"


def test_detection_share():
    frames = [
        {"hands": [{"label": "Left", "interpolated": False}]},
        {"hands": [{"label": "Left", "interpolated": True}]},
    ]
    s = detection_share(frames, detected_only=True)
    assert abs(s["left"] - 0.5) < 1e-9


def test_reentry_within_ten_frames():
    box = (0.2, 0.2, 0.5, 0.6)
    tracks, _out = assign_tracks([], [_det("Left", box)])
    for _ in range(10):
        tracks, out = assign_tracks(tracks, [])
    assert out[0]["identity"] == "Left"
    assert out[0]["interpolated"] is True
    _tracks, out = assign_tracks(tracks, [_det("Right", (0.22, 0.22, 0.52, 0.62))])
    assert out[0]["identity"] == "Left"
    assert out[0]["interpolated"] is False


def test_eleven_empty_frames_drop_the_track():
    box = (0.2, 0.2, 0.5, 0.6)
    tracks, _out = assign_tracks([], [_det("Left", box)])
    out = []
    for _ in range(11):
        tracks, out = assign_tracks(tracks, [])
    assert out == []
