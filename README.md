# How to Run:

## Part A: python main.py --mode detect

## Part B: python main.py --mode ocr --image sample_box.jpg

## Streamlit: streamlit run app.py


# Dataset & Model Selection Justification

## Dataset Selection (Part A)

### Detection Dataset (Fine-tuning Optional)

**Datasets Used:**

- Open Images V6 (Human + Animal subsets)
- Custom scraped wildlife datasets (optional)

### Why Not COCO or ImageNet?

| Reason | Explanation |
|-------|------------|
| Avoid leakage | COCO is heavily used in benchmarks and pretrained pipelines |
| Industrial realism | Custom datasets better match real deployment conditions |
| Better domain relevance | Real camera angles, lighting, and occlusions |

---

## Model Selection Justification

### Object Detection Model — Faster R-CNN (ResNet50-FPN)

**Reason for Selection:**

| Advantage | Reason |
|----------|--------|
| High accuracy | Suitable for safety-critical applications |
| Two-stage detector | Better localization compared to single-stage models |
| Offline friendly | No cloud dependency required |
| Stable backbone | Production-proven architecture |

---

### Classification Model — ResNet18 CNN

**Reason for Selection:**

| Advantage | Reason |
|----------|--------|
| Lightweight | Low-latency inference |
| Transfer learning | Pretrained weights improve convergence speed |
| Binary classification | Well suited for Human vs Animal classification |

---

### OCR Engine — Tesseract (Offline)

**Reason for Selection:**

| Advantage | Reason |
|----------|--------|
| Offline engine | No internet dependency |
| Industrial adoption | Widely used in scanning systems |
| Custom preprocessing | Highly tunable pipeline |
| Lightweight | CPU-friendly inference |

---

# Training Explanation (Part A)

## Object Detection Training

### Training Steps

1. Dataset annotation in **Pascal VOC / COCO format**
2. Image resizing to **640 × 640**
3. Data augmentation:
   - Horizontal flip
   - Random brightness
   - Scaling
4. Fine-tuning Faster R-CNN:
   - Backbone frozen initially
   - Detection head trained
   - Full network unfrozen in later stages

### Metrics Logged

- Training loss
- Validation mAP
- Precision
- Recall

---

## Classification Model Training

### Training Steps

1. Cropped bounding box images used as input
2. Label encoding:
   - `0 = Human`
   - `1 = Animal`
3. Preprocessing:
   - Resize to **224 × 224**
   - Normalize using **ImageNet statistics**
4. Training configuration:
   - Loss Function: Cross Entropy
   - Optimizer: Adam
   - Early stopping enabled

### Metrics Tracked

- Accuracy
- Validation loss

---

# Inference Pipeline Explanation

## Video Pipeline (Part A)

Video Input
↓
Frame Extraction
↓
Object Detection
↓
Bounding Box Cropping
↓
Classification
↓
Draw Bounding Boxes
↓
Save Output Video


### Performance Optimizations

- Frame skipping (2× speed improvement)
- Resolution downscaling
- Batch GPU inference (if available)

---

## OCR Pipeline (Part B)

Input Image
↓
Grayscale OCR
OR
Enhanced OCR
↓
Best Output Selection
↓
Text Cleaning Layer
↓
Structured JSON Output


---

# Challenges Faced

## Detection Speed Bottleneck

### Problem

- Faster R-CNN inference slow on CPU systems

### Solution

- Frame skipping
- Resolution scaling
- Batch inference

---

## OCR Noise Issues

### Problem

- Faded paint text
- Confusion with printed receipts

### Solution

- Dual OCR strategy
- Adaptive thresholding
- Post-processing cleanup rules

---

## Lighting Variations

### Problem

- Overexposed frames affecting detection

### Solution

- CLAHE (Contrast Limited Adaptive Histogram Equalization)

---

# Possible Improvements

## Detection Enhancements

- Integrate **ByteTrack** for multi-object tracking
- Apply **TensorRT optimization**
- Enable **quantized inference**
- Use lighter backbone for edge deployment

---

## OCR Enhancements

- Add **EAST text detector + region-wise OCR**
- Integrate **Transformer OCR (TrOCR offline)**
- Language model-based text correction
- Confidence score filtering
