# Training a custom 5-class YOLO model

This folder builds a YOLO model that detects **face, eyes and mouth natively**,
including whether the eyes are open or closed and whether the mouth is yawning.

## Why this is needed

The stock Ultralytics models (`yolov8n.pt`, `yolo11n.pt`) are trained on **COCO**,
whose 80 classes contain `person` — but no `face`, `eye` or `mouth`. A stock
model cannot produce the boxes this project needs.

The app ships with a real face-tuned checkpoint (one class: `face`) and gets eyes
and mouth from MediaPipe FaceMesh. That works well. Training your own model gives
you something extra: YOLO detects all five regions in a **single forward pass**,
so the system no longer depends on MediaPipe at runtime.

## The 5 classes

| id | name | meaning |
|----|--------------|------------------------------------|
| 0 | `face` | the whole face |
| 1 | `eye_open` | an eye that is open |
| 2 | `eye_closed` | an eye that is closed |
| 3 | `mouth_open` | mouth wide open (a yawn) |
| 4 | `mouth_closed` | mouth closed or only slightly open |

These ids are fixed in `data.yaml` and are read back by `src/detector.py`. Do not
reorder them.

---

## Dataset format

YOLO's "detect" format. Each image has a `.txt` label file with the **same
basename**:

```
training/dataset/
├── images/
│   ├── train/   normal_00001.jpg  ...
│   ├── val/     normal_00101.jpg  ...
│   └── test/    normal_00151.jpg  ...
└── labels/
    ├── train/   normal_00001.txt  ...
    ├── val/     normal_00101.txt  ...
    └── test/    normal_00151.txt  ...
```

Every label line is one object, in **normalised** coordinates (0–1):

```
<class_id> <x_center> <y_center> <width> <height>
```

A real example — a face, two open eyes, a closed mouth:

```
0 0.576562 0.418750 0.262500 0.466667
1 0.627344 0.359375 0.067187 0.039583
1 0.508594 0.368750 0.064062 0.037500
4 0.571094 0.551042 0.126562 0.085417
```

Coordinates are the box **centre** plus its size, divided by the image
dimensions. Normalising means the labels stay correct at any resolution.
An image with no objects has an empty `.txt` file.

---

## Step 1 — Record a labelled dataset

Hand-labelling thousands of boxes would take days. Instead we **auto-label**:
the pretrained YOLO-face model draws the face box, MediaPipe FaceMesh gives
exact eye and mouth contours, and the same EAR/MAR maths the live app uses
decides `open` vs `closed`.

This is weak supervision — a slow, complex teacher (YOLO-face + FaceMesh)
generates labels for a fast, simple student (one 5-class YOLO).

**Record yourself alone in the frame**, several times, varying the conditions:

```bash
python training/auto_label.py --session normal     --frames 400
python training/auto_label.py --session blinking   --frames 400
python training/auto_label.py --session yawning    --frames 300
python training/auto_label.py --session eyesshut   --frames 300
python training/auto_label.py --session glasses    --frames 300
python training/auto_label.py --session dim_light  --frames 300
python training/auto_label.py --session side_angle --frames 300
```

During each session, **act out that behaviour**: for `eyesshut`, keep your eyes
closed most of the time; for `yawning`, yawn repeatedly.

Notes:
- Frames with **no face** or **more than one face** are skipped automatically.
  A second person's eyes would be left unlabelled, and in YOLO training
  "unlabelled" means "background" — that quietly poisons the dataset.
- `--every 3` keeps 1 frame in 3. Consecutive webcam frames are nearly
  identical and teach the model nothing while inflating the dataset.

**Aim for 1500–3000 frames total.** Variety matters far more than volume: 1500
varied frames beat 5000 of you sitting still in the same light.

## Step 2 — Check the labels

```bash
python training/auto_label.py --preview --preview-count 12
```

This draws the saved labels back onto the saved images in
`training/dataset/preview/`. Open a few and confirm the boxes sit on the right
features and that open/closed is correct. **Never train on labels you have not
looked at.**

## Step 3 — Split into train / val / test

```bash
python training/auto_label.py --split
```

Default split is 75% train / 15% val / 10% test. It also prints the **class
balance**:

```
Class balance (number of boxes):
  face             1840  ##############################
  eye_open         2910  ##############################
  eye_closed        770  ############
  mouth_open        410  ######
  mouth_closed     1430  #######################
```

If a class is badly under-represented, the model will simply never predict it.
Record more sessions acting out that state and re-split.

## Step 4 — Train

```bash
python training/train.py --epochs 100 --batch 16
```

Useful options:

| Flag | Default | Notes |
|---|---|---|
| `--model` | `yolo11n.pt` | `n` nano (fastest) · `s`/`m` more accurate, slower |
| `--epochs` | `100` | fewer for a quick test |
| `--batch` | `16` | lower it if you run out of memory |
| `--imgsz` | `640` | training resolution |
| `--device` | `auto` | `mps` on Apple Silicon, `cuda`, or `cpu` |
| `--install` | off | copies the best weights into `models/` when done |

This is **transfer learning**: we start from `yolo11n.pt`, which already learned
generic visual features from COCO's 118k images, and re-teach only the final
layers. That is why a few thousand of your own frames is enough — training from
scratch would need hundreds of thousands.

**Augmentation** (configured in `train.py`) randomly alters each training image
so the model sees more variety than you recorded:

| Setting | Why |
|---|---|
| `hsv_v=0.4` | brightness — day, dusk, tunnel, night |
| `hsv_s=0.7` | saturation — different camera sensors |
| `degrees=10` | small rotation — heads tilt, they do not invert |
| `scale=0.4` | closer to / further from the camera |
| `fliplr=0.5` | mirroring — both eyes must be learned |
| `flipud=0.0` | **off** — an upside-down driver is not a real case |
| `erasing=0.2` | random blanked patches — mimics sunglasses/occlusion |
| `mosaic=1.0` | stitches 4 images together; strong regulariser |

On an M2, 100 epochs over ~2000 frames takes roughly **25–40 minutes**.

## Step 5 — Evaluate

```bash
python training/validate.py --weights training/runs/drowsiness/weights/best.pt
```

What the numbers mean:

- **Precision** — of everything called `eye_closed`, how much really was. Low
  precision means false alarms.
- **Recall** — of all real `eye_closed` boxes, how many were found. Low recall
  means **missed micro-sleeps**.
- **mAP50** — the standard detection score; a box counts as correct at ≥50% overlap.
- **mAP50-95** — much stricter, also rewards tight boxes.

For this application **recall on `eye_closed` matters most**: a false alarm is
annoying, a missed micro-sleep is a crash. The script flags it explicitly and
warns if it falls below 0.90.

## Step 6 — Use it

```bash
python training/train.py --epochs 100 --install
```

Then in `config/config.yaml`:

```yaml
model:
  weights: "models/drowsiness-yolo.pt"
```

The app inspects the model's class list on load. When it finds `eye_open` /
`eye_closed`, it switches to the native YOLO path automatically — no code change.
You can confirm this in the dashboard's **System information** panel:
`native_eye_detection: true`.

---

## Troubleshooting

**"Dataset images not found"** — Ultralytics resolves `path:` against a *global*
`datasets_dir` setting, not against `data.yaml`. `_dataset_path.py` works around
this by writing `.data.resolved.yaml` with an absolute path. If you invoke
`yolo` directly from the CLI, pass that resolved file.

**Loss becomes `nan` on MPS** — lower the learning rate (`--lr 0.005`) or train
on CPU (`--device cpu`).

**Model never predicts `eye_closed`** — check the class balance from step 3 and
record more `eyesshut` sessions.

**Out of memory** — lower `--batch` to 8 or 4, or `--imgsz` to 480.
