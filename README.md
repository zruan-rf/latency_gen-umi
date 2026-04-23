# Camera latency characterization

## venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data recording

In one window run `main.py` to start camera capturing. In another window run `ur_sine.py` to move the joint.

## Post processing

Put the recorded session folder at the repo root (same level as this README).

### AprilTag batch (head, left, or right)

Single script: `batch_apriltag_video.py`. At the top of the file set:

- **`SESSION_DIR_NAME`** — folder name for the run (e.g. `20260422_201653`).
- **`STREAM`** — `"head"`, `"left"`, or `"right"`.
- **`LEFT_CAMERA_ID`** / **`RIGHT_CAMERA_ID`** — for `STREAM == "left"` or `"right"`; must match `camera_id` in `capture_log.csv` (e.g. `left_video_0`, `right_video_1`).

Outputs are written **inside** that session folder:

| `STREAM` | Inputs | Outputs |
|----------|--------|---------|
| `head` | `SESSION/images/head/*.jpg`, `SESSION/realsense_log.csv` | `head_apriltag_processed.mp4`, `head_apriltag_processed.csv` |
| `left` | `SESSION/capture_log.csv` + `filename` paths for `LEFT_CAMERA_ID` | `left_apriltag_processed.mp4`, `left_apriltag_processed.csv` |
| `right` | same, for `RIGHT_CAMERA_ID` | `right_apriltag_processed.mp4`, `right_apriltag_processed.csv` |

```bash
python3 batch_apriltag_video.py
```

### Latency notebook

Open `latency_shift.ipynb`. In the second cell set **`SESSION`**, **`STREAM`** (`head` / `left` / `right`), and optional **`ROI_MONO_MIN`** / **`ROI_MONO_MAX`**. It loads the matching apriltag CSV + joint log, exports `SESSION/merged_output_{STREAM}.csv`, plots angles (matplotlib **`Agg`** backend, set before `pyplot` import to avoid GUI backend hangs), then estimates lag from **derivative zero-crossings** (not cross-correlation).
