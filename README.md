# VisDrone Object Detection — YOLOv26n

Drone-view object detection on the [VisDrone 2019](https://github.com/VisDrone/VisDrone-Dataset) dataset using YOLOv26n, trained from scratch on Kaggle.

---

## Training Setup

| Parameter | Value |
|-----------|-------|
| Model | YOLOv26n (2.37M params, 5.2 GFLOPs) |
| Dataset | VisDrone 2019 (10 classes) |
| Image size | 640 × 640 |
| Epochs | 50 |
| Batch size | default |
| Hardware | Kaggle — Tesla T4 × 2 |
| Training time | ~2.1 hours |
| Augmentation | Mosaic, Albumentations (blur, grayscale, CLAHE) |

---

## Results

| Metric | Value |
|--------|-------|
| mAP@50 | 0.334 |
| mAP@50-95 | 0.189 |
| Precision | 0.44 |
| Recall | 0.336 |

**Per-class mAP@50**

| Class | mAP@50 |
|-------|--------|
| car | 0.752 |
| bus | 0.437 |
| motor | 0.384 |
| pedestrian | 0.379 |
| van | 0.364 |
| truck | 0.310 |
| people | 0.294 |
| tricycle | 0.215 |
| awning-tricycle | 0.126 |
| bicycle | 0.082 |

---

## Key Findings

- **Severe class imbalance (~45:1)** — `car` has 144k samples vs `awning-tricycle` at 3.2k
- Performance strongly correlates with training sample count
- Model was still improving at epoch 50 — more epochs will help
- Small/rare objects remain the core challenge on drone imagery

---

## Sample Predictions

<p float="left">
  <img src="results/predictions/9999939_00000_d_0000166.jpg" width="32%" />
  <img src="results/predictions/9999939_00000_d_0000196.jpg" width="32%" />
  <img src="results/predictions/9999939_00000_d_0000271.jpg" width="32%" />
</p>

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Inference

```python
from ultralytics import YOLO

model = YOLO("model/best.pt")
results = model.predict("image.jpg", conf=0.25, imgsz=640, save=True)
```

---

## Project Structure

```
model/          # best.pt weights
config/         # dataset YAML
results/        # training logs, prediction images
notebook/       # training + analysis notebook
```

---

## What's Next

- Train longer (100 epochs) — model hadn't converged
- Upgrade to YOLOv26s/m for more capacity
- Use `imgsz=1280` for better small object detection
- Apply SAHI slice inference for rare/tiny classes
