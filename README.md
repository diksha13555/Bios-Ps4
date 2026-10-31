# TouchTrace / PS4 — egocentric hand tracking (BIOS V2, Problem Statement 2)

Track left and right hands with MediaPipe Hand Landmarker (VIDEO mode), stabilize identity across frames with IoU and center-distance matching, smooth landmarks with One Euro filtering, and export per-frame JSONL plus operational metrics. Optional side-by-side visualization and sim export remain unchanged.

Plan: [Notion — PS4 (24h)](https://www.notion.so/40aa9a9820cb818dbed3e37e150f2161)

## Data rules

Hackathon use only. Do not publish clips, frames, or screenshots. Do not commit video. `data/raw/` and `outputs/` are gitignored.

## Installation (Windows)

```powershell
cd C:\Users\Loq\OneDrive\Desktop\Bios-Ps4
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Download the MediaPipe model (Git Bash or WSL):

```bash
bash scripts/download_models.sh
```

Or download manually to `models/hand_landmarker.task` from the URL in `scripts/download_models.sh`.

Run tests:

```powershell
pytest -q
```

## Run hand-tracking pipeline

```powershell
python scripts/run_ps4.py --input data/raw/your_clip.mp4
```

| Output | Description |
| --- | --- |
| `<name>_sidebyside.mp4` | Source + 2D skeleton beside 3D hand plot |
| `<name>.jsonl` | Per-frame landmarks (image, world, optional camera) |
| `<name>_report.json` | **Operational** metrics: FPS, detection share, dropouts, jitter proxy |
| `<name>_sim.json` | Wrist 6-DoF and finger curl angles for sim |

Useful flags: `--max-frames 300`, `--no-clahe`, `--viz-max-side 960`, `--no-camera-pose`, `--max-gap 5`.

## JSONL output format

One JSON object per line:

```json
{
  "frame": 0,
  "t_sec": 0.0,
  "hands": [
    {
      "label": "Left",
      "score": 0.91,
      "tracked": true,
      "interpolated": false,
      "image": [[x, y, z], "... 21 landmarks ..."],
      "world": [[x, y, z], "... 21 landmarks ..."],
      "camera": null,
      "reproj_error_px": null
    }
  ]
}
```

Coordinates: `image` x/y are normalized to frame width/height in `[0, 1]`.

## Ground-truth annotation format

Label only frames you have manually annotated. Omit all other frames (they are excluded from accuracy metrics).

```json
{"frame": 12, "hands": [{"label": "Left", "image": [[0.42, 0.31, 0.0], "... 21 points ..."]}]}
{"frame": 13, "hands": []}
```

- `label`: `"Left"` or `"Right"` (annotator’s subject handedness).
- `image`: 21 landmarks in the same normalized space as pipeline output (z optional).
- Empty `hands`: negative example (no hand visible).

Do not generate labels from model output.

## Accuracy evaluation

```powershell
python scripts/evaluate_hands.py `
  --predictions outputs/your_clip.jsonl `
  --labels data/labels/your_clip.jsonl `
  --operational-report outputs/your_clip_report.json `
  --report outputs/your_clip_eval.json
```

### Metrics

| Block | Metric | Meaning |
| --- | --- | --- |
| **detection** | precision / recall / detection_rate | Spatial wrist matching; **labels ignored** for TP/FP/FN |
| **landmarks** | mean_error_norm | Mean L2 error on x,y for matched pairs with 21 GT points |
| **landmarks** | pck | Fraction of keypoints within `--pck-threshold` (default 0.05 norm) |
| **handedness** | accuracy | Label agreement on **spatially matched** pairs only |
| **tracking** | identity_label_switches | Prediction label changes for the same GT hand across labelled frames |
| **operational** | (from `_report.json`) | FPS, detection share, dropouts, mean confidence, world speed — **not** ground-truth accuracy |

If no annotation file exists, the evaluator exits with a clear message and does not report fabricated accuracy.

## Pipeline architecture

MediaPipe Hand Landmarker (VIDEO + IMAGE crop retry), CLAHE preprocessing, IoU identity tracking with center-distance fallback for fast motion, One Euro smoothing, gap fill up to 5 frames on export, optional `solvePnP` camera landmarks.

**Mirrored labels:** `src/mirrored_label.py` is a documented hook. The hackathon PDF is not in-repo; **no Left/Right swap is applied** until the spec is confirmed.

## Limitations

- No labelled evaluation video is bundled; real accuracy numbers require your annotation JSONL.
- Live track hold (`HandRuntime`) uses `max_gap=10`; export gap fill uses CLI `--max-gap` (default 5).
- MediaPipe world landmarks are hand-centric meters, not a factory frame.
- End-to-end video pipeline was not run in CI without `data/raw/*.mp4` and `models/hand_landmarker.task`.

## Test results (last run)

```
18 passed in 0.85s
```

Includes identity tracking, smoothing, gap fill, JSONL schema, mirrored-label stub, and ground-truth evaluation (synthetic fixtures only). Full video pipeline not run (no sample MP4 or model in repo).
