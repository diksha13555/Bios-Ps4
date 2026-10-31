import json

from src.hands import to_public_hand
from src.pipeline import _json_frame


def test_public_hand_and_jsonl_shape():
    internal = {
        "identity": "Left",
        "score": 0.88,
        "image": [[0.1, 0.2, 0.0]] * 21,
        "world": [[0.01, 0.02, 0.03]] * 21,
        "interpolated": False,
    }
    pub = to_public_hand(internal)
    assert pub["label"] == "Left"
    assert len(pub["image"]) == 21
    frame = _json_frame({"frame": 0, "t_sec": 0.0, "hands": [pub]})
    line = json.loads(json.dumps(frame))
    assert line["frame"] == 0
    assert line["hands"][0]["label"] in ("Left", "Right")
    assert "tracked" in line["hands"][0]
    assert "interpolated" in line["hands"][0]
