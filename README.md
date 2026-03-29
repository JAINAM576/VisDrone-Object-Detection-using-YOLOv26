# 🚁 VisDrone Object Detection (YOLOv26)

Object detection on the VisDrone dataset using YOLOv26, with focus on training analysis, class imbalance, and real-world inference.

---

## 📊 Results

- mAP50: **0.33**
- mAP50-95: **0.18**
- Strong performance on **cars**
- Lower performance on **small / rare objects**

---

## 🔍 Key Insights

- Dataset has **severe class imbalance (~45:1)**
- Model performs better on **frequent classes**
- Small object detection remains challenging

---

## 🖼️ Sample Predictions

![Prediction](results/predictions/sample1.jpg)

---

## ⚙️ Setup

```bash
pip install ultralytics
````

---

## 🚀 Inference

```python
from ultralytics import YOLO

model = YOLO("model/best.pt")
results = model("image.jpg", save=True)
```

---

## 📁 Project Structure

```
model/          # trained weights
config/         # dataset yaml
results/        # logs + predictions
notebook/       # training notebook
```

---

## 📌 Conclusion

* Model learns well but is limited by **data imbalance**
* Increasing image size and better sampling can improve results

