# Driver Drowsiness Detection — Real-Time, YOLO + Webcam

A real-time driver monitoring system. Point a webcam at a driver and it detects
the face with **YOLO**, measures eyelid and mouth movement with **MediaPipe
FaceMesh**, computes a continuous **drowsiness score**, draws an **attention
heatmap**, and **sounds an alarm** when the driver is falling asleep.

It runs entirely on the live camera. There is no image upload, no video upload,
and no manual input anywhere in the system.

---

## What it does

| Capability | How |
|---|---|
| Face detection | YOLO11n fine-tuned on WIDER FACE — real boxes, real confidence |
| Eye / mouth regions | MediaPipe FaceMesh landmarks inside the YOLO face box (or native YOLO classes with the custom model) |
| Eye closure | Eye Aspect Ratio (EAR) per eye |
| Blink counting | Closures shorter than 0.4 s |
| Micro-sleep detection | Closures longer than 1.2 s → instant alarm |
| Yawn detection | Mouth Aspect Ratio (MAR) held above threshold for 0.8 s |
| Drowsiness score | Weighted blend of PERCLOS, current closure, yawn rate, blink rate |
| Attention heatmap | Gaussian blobs over eyes/mouth scaled by live EAR/MAR |
| Alarm | Synthesised two-tone siren, played off-thread with a cooldown |
| Dashboard | Streamlit: live feed, heatmap, all metrics, FPS, score history |

---

## Requirements

- **Python 3.10, 3.11 or 3.12** (MediaPipe has no wheels for 3.13+ — use 3.11)
- A webcam
- macOS (Apple Silicon or Intel), Linux, or Windows

---

## Installation

```bash
# 1. Go to the project folder
cd /Users/sasidharreddy/Desktop/DV

# 2. Create a virtual environment with Python 3.11
python3.11 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\activate           # Windows

# 3. Install the dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 4. Download the YOLO face-detection weights
python tools/download_models.py

# 5. Confirm everything works on this machine
python tests/smoke_test.py
```

`tests/smoke_test.py` checks the config, the model, MediaPipe, the alarm, the
webcam, the live pipeline and the dashboard script. Every line should say
`PASS` before you continue.

---

## Running

### The dashboard (main application)

```bash
streamlit run app.py
```

Your browser opens at `http://localhost:8501`. **The webcam starts by itself** —
you do not click anything. Sit in front of the camera and the boxes, score and
heatmap appear immediately.

### The OpenCV window version (fastest, best for a live demo)

```bash
python run_cli.py
```

Keys: `q` quit · `r` reset counters · `h` heatmap · `b` boxes · `l` landmarks ·
`space` test alarm · `s` screenshot.

---

## macOS camera permission

The first run triggers a permission prompt. If you miss it or deny it, the app
shows a clear error instead of a blank screen.

1. **System Settings → Privacy & Security → Camera**
2. Enable the app you launched from — **Terminal**, **iTerm**, or your IDE
3. **Fully quit that app and reopen it.** macOS only re-reads the permission on
   a fresh launch; restarting only the Python process is not enough.

---

## How to test it (what to show a demo audience)

| To trigger | Do this | Expected |
|---|---|---|
| Blink counter | Blink normally | `Blinks` increases, score stays low |
| Eye closure | Close your eyes and hold | `Eye closure` timer climbs, heatmap turns red over the eyes |
| **Alarm** | Keep eyes closed past ~1.2 s | Status flips to **DROWSY**, red border, alarm sounds |
| Yawn | Open your mouth wide for ~1 s | `Yawns` increases, mouth region turns red |
| No driver | Step out of frame | Status shows **NO FACE**, score decays |
| Multiple faces | Have someone lean in | Largest face is labelled `DRIVER`, others `passenger` |

---

## Project structure

```
DV/
├── app.py                    Streamlit dashboard (main entry point)
├── run_cli.py                OpenCV-window version
├── requirements.txt
├── README.md
│
├── config/
│   └── config.yaml           Every tunable threshold and weight
│
├── src/
│   ├── config.py             Loads and validates config.yaml
│   ├── camera.py             Threaded webcam capture + error handling
│   ├── detector.py           YOLO wrapper, device selection, box parsing
│   ├── landmarks.py          MediaPipe FaceMesh → EAR / MAR / region boxes
│   ├── drowsiness.py         The scoring algorithm (state machine)
│   ├── heatmap.py            Gaussian attention map generation
│   ├── alarm.py              Sound synthesis + non-blocking playback
│   ├── overlay.py            All drawing: boxes, labels, HUD
│   └── pipeline.py           Ties every stage together, one frame at a time
│
├── models/
│   └── yolov11n-face.pt      Downloaded by tools/download_models.py
│
├── training/                 Custom 5-class YOLO model (see training/README.md)
│   ├── data.yaml             Dataset definition and class list
│   ├── auto_label.py         Builds a labelled dataset from your webcam
│   ├── train.py              Fine-tunes YOLO on it
│   └── validate.py           Precision / recall / mAP on the test split
│
├── tools/download_models.py  Fetches the pretrained YOLO face weights
├── tests/smoke_test.py       End-to-end verification
└── docs/explanation.html     Full illustrated write-up of how it works
```

---

## How the algorithm works

### 1. Eye Aspect Ratio (EAR)

Six landmarks around each eye:

```
      p2   p3
  p1            p4      EAR = (|p2−p6| + |p3−p5|) / (2 · |p1−p4|)
      p6   p5
```

The numerator is the eyelid gap; the denominator is the eye width. Dividing by
the width makes EAR **scale-invariant** — leaning toward the camera does not
change it.

- Open eye → **0.25 – 0.35**
- Closed eye → **below 0.21** (the configured threshold)

### 2. Blink vs micro-sleep

Both are "eyes closed". They are separated by **duration**, measured when the
eyes reopen:

- `≤ 0.40 s` → a normal **blink** (counted, harmless)
- `≥ 1.20 s` → a **micro-sleep** → forces `DROWSY` immediately

### 3. Yawning (MAR)

Mouth Aspect Ratio = lip gap ÷ mouth width. A yawn is a *sustained* high MAR, so
the mouth must stay open for 0.8 s. This is what stops normal talking from being
counted as yawning.

### 4. The drowsiness score

Four independent channels, each normalised to 0–1, then weighted:

```
score = 100 × ( 0.40·PERCLOS + 0.35·closure + 0.15·yawn + 0.10·blink )
```

| Channel | Weight | Meaning |
|---|---|---|
| PERCLOS | 40% | Fraction of time eyes were closed over the last 60 s. The measure that best correlates with real fatigue in driving research. |
| Closure | 35% | How long the eyes are closed *right now*. Catches a micro-sleep within ~1 s. |
| Yawn | 15% | Yawns per minute — an early warning sign. |
| Blink | 10% | Blink rate outside the normal 8–28/min band, in either direction. |

The result is exponentially smoothed so the needle does not flicker.

**Bands:** `< 40` ALERT · `40–59` WARNING · `≥ 60` DROWSY

**Safety override:** eyes closed longer than the micro-sleep threshold forces
DROWSY regardless of the score. At 100 km/h a car covers ~33 m per second —
there is no time to wait for a rolling average.

### 5. The attention heatmap

A 2D Gaussian is added at each eye and at the mouth:

```
G(x,y) = A · exp( −( (x−cx)² / 2σx²  +  (y−cy)² / 2σy² ) )
```

- **Amplitude `A`** comes from the live measurements — lower EAR means a hotter
  eye, higher MAR means a hotter mouth.
- **Width `σ`** comes from the size of the detected region.

The blobs are summed, colour-mapped, and blended with a *per-pixel* weight, so
cold regions stay perfectly sharp and only hot regions get tinted.

---

## Configuration

Everything lives in `config/config.yaml`. The values you are most likely to
change:

```yaml
drowsiness:
  ear_threshold: 0.21        # raise if closures are missed, lower for false alarms
  microsleep_duration: 1.20  # seconds of closure that force DROWSY
  mar_threshold: 0.60        # yawn sensitivity
  drowsy_score: 60           # alarm threshold

model:
  weights: "models/yolov11n-face.pt"
  confidence: 0.35
  imgsz: 480                 # lower = faster, higher = more accurate

camera:
  index: 0                   # try 1, 2… for an external camera
```

The dashboard sidebar lets you change the main thresholds live, without a
restart — useful for tuning in front of an audience.

---

## The YOLO model

### What ships by default

`models/yolov11n-face.pt` — a **YOLO11n checkpoint genuinely fine-tuned for face
detection** (Apache-2.0), downloaded from Hugging Face by
`tools/download_models.py`. It has one class: `face`.

### Why not stock YOLO?

The stock Ultralytics models (`yolov8n.pt`, `yolo11n.pt`) are trained on COCO,
whose 80 classes include `person` but **not** `face`, `eye` or `mouth`. A stock
model physically cannot produce the boxes this project needs. That is why we
either use a real face-tuned checkpoint or train our own.

### Where eyes and mouth come from

With the face-only model, YOLO locates the face and **MediaPipe FaceMesh**
locates the eyes and mouth inside that box, from 478 measured landmarks. Nothing
is hardcoded or assumed — the boxes are computed from the actual landmark
positions every frame.

### Training your own 5-class model

To have YOLO detect eyes and mouth **natively** (classes `face`, `eye_open`,
`eye_closed`, `mouth_open`, `mouth_closed`), see **[training/README.md](training/README.md)**.
The short version:

```bash
python training/auto_label.py --session normal   --frames 400
python training/auto_label.py --session eyesshut --frames 300
python training/auto_label.py --session yawning  --frames 300
python training/auto_label.py --split
python training/train.py --epochs 100 --install
```

Then set `weights: "models/drowsiness-yolo.pt"` in `config/config.yaml`. The app
detects the extra classes automatically and switches to the native path.

---

## Error handling

| Situation | What happens |
|---|---|
| Camera permission denied | Explicit macOS instructions, not a blank screen |
| No camera found | Cameras are probed; you are told which indices work |
| Camera unplugged mid-session | Detected after 30 failed reads, clean error, camera released |
| Camera busy (Zoom/Teams) | Reported as an open failure with a fix list |
| Model file missing | Names the exact command to download or train one |
| Corrupted weights | Suggests deleting and re-downloading |
| MPS/GPU inference fails | Silently falls back to CPU and keeps running |
| No face in frame | `NO FACE` state; score decays; counters preserved |
| Multiple faces | Largest = `DRIVER`, others drawn grey as `passenger` |
| Low-confidence detections | Filtered by the confidence threshold |
| Low FPS | FPS tile turns amber below 15; lower `model.imgsz` to recover |
| Audio backend missing | Falls back through pygame → aplay → terminal bell |

---

## Performance

Measured on an Apple M2, 640×480 input, `imgsz: 480`:

| Stage | Time |
|---|---|
| YOLO inference (MPS) | ~9 ms |
| MediaPipe FaceMesh | ~5 ms |
| Heatmap build + blend | ~5 ms |
| Drawing + HUD | ~2 ms |
| **End-to-end** | **~27 ms → 33–35 FPS** |

If you need more speed: lower `model.imgsz` to 320, set `heatmap.enabled: false`,
or reduce `camera.width`/`height`.

### Apple Silicon notes

- The device is auto-selected: `mps` on Apple Silicon, `cuda` on NVIDIA, else `cpu`.
- `half_precision` stays `false` — fp16 is not reliably stable on MPS.
- The camera uses the **AVFoundation** backend, which is required for good
  capture performance on macOS.
- If the first MPS inference fails, the detector permanently falls back to CPU
  rather than crashing.

---

## Troubleshooting

**"Camera opened but returned no usable frames"** — camera permission. See the
macOS section above, and remember to fully quit and reopen the terminal.

**`ModuleNotFoundError: mediapipe`** — you are on Python 3.13+. Recreate the
venv with `python3.11 -m venv .venv`.

**Score sits high while I am awake** — your EAR baseline differs from the
default. Watch the EAR tile with your eyes open, then set `ear_threshold` to
roughly 70% of that value.

**Yawns triggered by talking** — raise `mar_threshold` or `yawn_min_duration`.

**FPS below 15** — lower `model.imgsz` to 320 and/or disable the heatmap.

**Streamlit shows a stale frame after Stop** — press Start again; the camera is
re-opened fresh.

---

## Documentation

`docs/explanation.html` is a full illustrated write-up — architecture, the maths
behind EAR/MAR/PERCLOS, the scoring design, the training procedure, and the
design decisions with their justifications. Open it in any browser:

```bash
open docs/explanation.html
```

---

## Licence notes

- The code in this repository is yours to use.
- `models/yolov11n-face.pt` comes from
  [AdamCodd/YOLOv11n-face-detection](https://huggingface.co/AdamCodd/YOLOv11n-face-detection)
  (**Apache-2.0**).
- Ultralytics YOLO itself is **AGPL-3.0**. That matters if you distribute this
  as a product; it does not affect academic or personal use.
