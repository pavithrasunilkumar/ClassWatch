# ClassWatch — Privacy-Aware Collective Attention Analysis System

An advanced computer vision-driven system for analyzing collective classroom attention. It combines deep learning-based detection (YOLOv8), multi-object tracking (ByteTrack), and head pose estimation (MediaPipe) to generate temporal attention metrics. The system features real-time visualization and enforces privacy through anonymization via face blurring.

---

## Pipeline Architecture

The pipeline runs in a continuous loop per frame:

```
Webcam Input (640×480)
    │
    ▼
YOLOv8n (GPU) ──────────────────────────► privacy.py
    │                                       Gaussian blur on face ROI (in-memory)
    ▼
ByteTrack ──────────────────────────────► logger.py
    │                                       Time-gated CSV write every 2 seconds
    ▼
MediaPipe Face Mesh                         data/attention_log.csv
    │                                       data/student_log.csv
    ▼
Attention Logic  (|yaw| < 30°, |pitch| < 25°)
    │
    ▼
Smoothing  (majority vote, W=5 frames) ──► utils.py
    │                                       HUD overlay on OpenCV window
    ▼
dashboard.py  ─── SocketIO ──────────────► Browser (localhost:5000)
                                            • Attention gauge
                                            • Line chart (attention % over time)
                                            • Donut chart (attentive vs distracted)
                                            • Distraction event log with timestamps
                                            • Per-student sparkline cards

On press 'q':
    ├──► analytics.py → outputs/attention_graph.png
    │                   outputs/summary.txt
    └──► dashboard.py → session_summary SocketIO event
                        (peak time, low time, per-student averages)
```

---

## Features

- **Real-time person detection** — YOLOv8n on GPU, handles up to `YOLO_MAX_DET` students simultaneously
- **Persistent tracking** — ByteTrack assigns stable IDs across frames even with occlusion
- **Head-pose attention** — MediaPipe Face Mesh estimates yaw/pitch to classify Attentive vs Distracted
- **Privacy layer** — MediaPipe face detection + Gaussian blur applied per frame, no face images stored
- **Live web dashboard** — Flask + SocketIO streams data to browser in real time
  - Animated attention gauge
  - Attention % over time (line chart)
  - Attentive vs Distracted split (donut chart)
  - Distraction event log with exact timestamps
  - Per-student live history sparklines
- **Session summary** — pushed to browser on stop, includes peak/low attention times and per-student averages
- **CSV logging** — class-level and per-student logs written every 2 seconds
- **Post-session analytics** — attention graph image + summary text file auto-generated on exit

---

## Project Structure

```
project/
│
├── main.py            ← Entry point — runs the full pipeline
├── logger.py          ← Temporal CSV logging (time-gated, 2s intervals)
├── analytics.py       ← Graph generation + session statistics
├── dashboard.py       ← Flask web server + SocketIO + HTML dashboard
├── privacy.py         ← Face detection + Gaussian blur (no storage)
├── utils.py           ← Shared helpers: FPS counter, HUD drawing, etc.
├── architecture.svg   ← Pipeline architecture diagram
│
├── data/
│   ├── attention_log.csv     ← Class-level log (auto-created)
│   └── student_log.csv       ← Per-student state log (auto-created)
│
└── outputs/
    ├── attention_graph.png   ← Generated on session end
    └── summary.txt           ← Generated on session end
```

---

## Requirements

### Hardware
- Webcam (built-in or USB)
- NVIDIA GPU required for the detection pipeline. This setup is intended to run on CUDA.

### Python
Python 3.10 or 3.11 recommended. Python 3.13 is **not** supported — MediaPipe and PyTorch wheels are unavailable for 3.13.

### Install dependencies

```bash
pip install ultralytics mediapipe opencv-python matplotlib flask flask-socketio requests torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

> If you are using a virtual environment (recommended):
> ```bash
> python -m venv venv
> venv\Scripts\activate        # Windows
> source venv/bin/activate     # Mac / Linux
> pip install ultralytics mediapipe opencv-python matplotlib flask flask-socketio requests torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
> ```

### Upgrade MediaPipe (if needed for Python 3.12+)

If you encounter `MediaPipe unavailable` errors with Python 3.12 or higher:

```bash
pip uninstall mediapipe -y
pip install mediapipe --upgrade --no-cache-dir
```

---

## How to Run

**Step 1 — Activate your virtual environment**
```bash
venv\Scripts\activate
```

**Step 2 — Run the system**
```bash
python main.py
```

**Step 3 — Open the dashboard**

The browser opens automatically at:
```
http://localhost:5000
```

If it does not open, paste that URL into any browser manually.

**Step 4 — Stop the session**

Press `q` in the OpenCV camera window.

The system will:
1. Stop the camera feed
2. Compute session analytics
3. Push the full summary to the browser
4. Save `attention_graph.png` and `summary.txt` to `outputs/`

---

## Configuration

All tunable parameters are at the top of `main.py`:

```python
# ── Camera & display ──────────────────────────────────────
CAMERA_INDEX     = 0        # 0 = default webcam, 1 = external USB cam
FRAME_WIDTH      = 1280     # display resolution (detection runs at 640×480 internally)
FRAME_HEIGHT     = 720

# ── YOLO detection ────────────────────────────────────────
YOLO_MODEL       = "yolov8n.pt"   # yolov8n = fastest, yolov8m = more accurate
YOLO_CONF        = 0.4            # detection confidence threshold (0.0–1.0)
YOLO_DEVICE      = "cuda"         # required: CUDA-only mode
YOLO_MAX_DET     = 10             # max students tracked at once — raise for larger classes

# ── Attention smoothing ───────────────────────────────────
SMOOTHING_WINDOW = 5        # majority vote over last N frames (higher = smoother but slower)

# ── Privacy ───────────────────────────────────────────────
PRIVACY_ENABLED  = True     # set False to disable face blurring

# ── Head-pose thresholds ──────────────────────────────────
YAW_THRESHOLD    = 30       # degrees left/right before marked Distracted
PITCH_THRESHOLD  = 25       # degrees up/down before marked Distracted
```

### Tuning tips

| Situation | What to change |
|---|---|
| Missing students in detection | Lower `YOLO_CONF` to `0.3`, raise `YOLO_MAX_DET` |
| Too many false detections | Raise `YOLO_CONF` to `0.5` |
| Attention flickers too much | Raise `SMOOTHING_WINDOW` to `10` |
| Sideways face still attentive | Lower `YAW_THRESHOLD` to `20` |
| No GPU available | Install a CUDA-enabled PyTorch build for your GPU |
| External / USB camera | Set `CAMERA_INDEX = 1` |

---

## Dashboard Walkthrough

| Section | What it shows |
|---|---|
| **Live Attention gauge** | Current frame attention %, colour shifts green → yellow → red |
| **Session Average KPI** | Running average since session started |
| **Students Detected KPI** | Total tracked + how many are attentive right now |
| **Session Started KPI** | Start time + current FPS |
| **Attention % Over Time** | Scrolling line chart of last 30 log points |
| **Attentive vs Distracted** | Live donut chart split |
| **Distraction Events** | Table of exact timestamps when class dropped below 50% attention |
| **Per-Student History** | Card per student with sparkline bar history |
| **Session Summary** | Appears automatically when you press `q` — shows all stats + per-student averages |

---

## Output Files

After pressing `q`, check the `outputs/` folder:

| File | Contents |
|---|---|
| `outputs/attention_graph.png` | Dark-themed line chart with peak and low markers |
| `outputs/summary.txt` | Plain text summary: averages, stability score, student attentiveness |
| `data/attention_log.csv` | Timestamped class-level rows (every 2 seconds) |
| `data/student_log.csv` | Per-student state history |

---

## Common Errors

**`ImportError: cannot import name X from dashboard`**
Make sure you are using the latest `main.py` and `dashboard.py` from this project. Do not mix files from different versions.

**`Cannot open camera index 0`**
Another application is using the webcam. Close Teams, Zoom, or any other camera app and try again. If using an external camera, set `CAMERA_INDEX = 1`.

**`bytetrack.yaml not found`**
This file is bundled with `ultralytics`. Run `pip install --upgrade ultralytics` to ensure it is present.

**`Torch not compiled with CUDA enabled`**
The default `pip install torch` installs a CPU-only build. Install the CUDA build explicitly:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```
Or set `YOLO_DEVICE = "cpu"` in `main.py` to run on CPU (slower but functional).

**`module 'mediapipe' has no attribute 'solutions'`**
You are on Python 3.13 which MediaPipe does not support. Use Python 3.11:
```bash
py -3.11 -m venv venv
```

**Dashboard shows but no data arrives**
The SocketIO connection may be blocked. Ensure nothing else is running on port 5000. You can change `WEB_PORT` at the top of `dashboard.py`.

**Low FPS / laggy video**
Ensure your CUDA drivers and CUDA-enabled PyTorch build are installed correctly. Keep `YOLO_MODEL = "yolov8n.pt"` for the fastest model.

---

## Privacy Notes

- Face images are **never saved to disk** at any point
- Blurring is applied directly to the frame buffer in memory and discarded each frame
- CSV logs store only numeric counts and timestamps — no biometric data
- Student IDs are arbitrary integers assigned by ByteTrack and reset each session

---

## Built With

| Library | Purpose |
|---|---|
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Person detection |
| [ByteTrack](https://github.com/ifzhang/ByteTrack) | Multi-object tracking |
| [MediaPipe](https://mediapipe.dev) | Face detection + Face Mesh for head pose |
| [OpenCV](https://opencv.org) | Camera capture, drawing, display |
| [Flask + Flask-SocketIO](https://flask-socketio.readthedocs.io) | Web dashboard server |
| [Chart.js](https://www.chartjs.org) | Browser-side charts |
| [Matplotlib](https://matplotlib.org) | Post-session graph image |


ACADEMIC PROJECT -MIT WPU
