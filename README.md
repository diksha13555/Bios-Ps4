# PS4 — 3D hand motion from egocentric video

Track left and right hands with MediaPipe, smooth and stabilize identity across frames, and render a side-by-side video: source with a 2D skeleton plus a 3D panel from world (or approximate camera) landmarks. Exports per-frame JSONL, a metrics report, and sim-friendly wrist pose and finger curl angles.

Plan: [Notion — PS4 (24h)](https://www.notion.so/40aa9a9820cb818dbed3e37e150f2161)

## Data rules

Hackathon use only. Do not publish clips, frames, or screenshots. Do not commit video. `data/raw/` and `outputs/` are gitignored. Processed demos stay on a private Drive folder.

## Setup

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
bash scripts/download_models.sh
pytest -q
```

## Run

```bash
python scripts/run_ps4.py --input data/raw/your_clip.mp4
```

| Output | Description |
| --- | --- |
| `<name>_sidebyside.mp4` | Source + 2D skeleton beside 3D hand plot |
| `<name>.jsonl` | Per-frame landmarks (image, world, optional camera) |
| `<name>_report.json` | Detection share, **`accuracy`** block (confidence, coverage, pitch_lead), FPS |
| `<name>_sim.json` | Wrist 6-DoF and finger curl angles for sim |

Useful flags: `--max-frames 300`, `--no-clahe`, `--viz-max-side 960`, `--no-camera-pose`, `--max-gap 5`, `--min-detection-confidence 0.4` (recall vs precision).

## Pipeline

MediaPipe Hand Landmarker (VIDEO mode) with CLAHE preprocessing, IoU-based left/right tracking, One Euro smoothing, gap fill up to 5 frames, optional `solvePnP` camera-frame landmarks, and Matplotlib 3D rendering.
