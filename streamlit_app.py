"""Streamlit Community Cloud entrypoint (main file: streamlit_app.py)."""

from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
MODEL_PATH = ROOT / "models" / "hand_landmarker.task"


def _bootstrap() -> None:
    for rel in ("data/raw", "outputs", "models"):
        (ROOT / rel).mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.is_file():
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)


_bootstrap()

from app.workbench import main

main()
