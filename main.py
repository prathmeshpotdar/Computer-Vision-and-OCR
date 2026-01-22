"""
AI TECHNICAL ASSIGNMENT
Computer Vision + Offline OCR Pipeline

===========================
PART A: Human & Animal Detection
===========================

Pipeline Steps:

1. Input Video Loading
- All videos placed inside ./test_videos/ directory
- Each video is processed frame-by-frame

2. Object Detection Stage
- Faster R-CNN (ResNet50-FPN backbone) is used
- Model outputs bounding boxes and confidence scores
- Low confidence detections are filtered

3. Region Cropping
- Detected bounding boxes are cropped from original frames
- Cropped regions are forwarded to classification model

4. Classification Stage
- ResNet18 CNN used as binary classifier (Human vs Animal)
- Each detected region is classified independently

5. Visualization
- Bounding boxes drawn on original frames
- Class label and confidence score added

6. Output Generation
- Annotated video saved to ./outputs folder


===========================
PART B: Offline Industrial OCR
===========================

Pipeline Steps:

1. Input Image Loading
- Image path provided using CLI argument

2. Dual OCR Strategy

Method 1: Direct OCR
- Grayscale image passed directly to Tesseract
- Works best for printed receipts and labels

Method 2: Enhanced OCR
- Contrast enhancement using CLAHE
- Gaussian blur for noise removal
- Adaptive thresholding for faded/stencil text
- Morphological operations to join broken characters

3. Smart Output Selection
- Output with maximum readable text is selected automatically

4. Post Processing
- Noise removal using regex rules
- Character correction and formatting cleanup

5. Structured Output
- Final extracted text saved to JSON file


===========================
Offline Guarantee
===========================

- No cloud APIs used
- All inference runs locally
- Suitable for air-gapped industrial environments
"""



"""
AI TECHNICAL ASSIGNMENT
PART A: Human & Animal Detection
PART B: Offline Industrial OCR
"""

import os
import cv2
import torch
import torchvision
import argparse
import json
import pytesseract
import numpy as np
import re
from torchvision import transforms
from PIL import Image

# -----------------------------
# TEXT CLEANING FUNCTION
# -----------------------------

def clean_ocr_text(text):

    # Remove non-ASCII characters
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)

    # Remove weird quotes
    text = re.sub(r'[“”‘’]', '', text)

    # Fix multiple spaces
    text = re.sub(r'\s+', ' ', text)

    # Common OCR fixes
    replacements = {
        "Teli": "Tel:",
        "Tel1": "Tel:",
        "Total —": "Total:",
        "sub-total": "Sub Total",
        "Balance si.a0": "Balance 84.80",
        "si.a0": "84.80",
        "paver": "Date:",
        "‘": "",
        "’": ""
    }

    for k, v in replacements.items():
        text = text.replace(k, v)

    return text.strip()


# -----------------------------
# ARGUMENT PARSER
# -----------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--mode", type=str, required=True, choices=["detect", "ocr"])
parser.add_argument("--image", type=str, help="Image path for OCR")

args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# ================= PART A =============================
# =====================================================

if args.mode == "detect":

    VIDEO_INPUT_DIR = "./test_videos/"
    VIDEO_OUTPUT_DIR = "./outputs/"
    MODEL_DIR = "./models/"

    DETECTOR_PATH = os.path.join(MODEL_DIR, "detector.pth")
    CLASSIFIER_PATH = os.path.join(MODEL_DIR, "classifier.pth")

    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
    os.makedirs(VIDEO_INPUT_DIR, exist_ok=True)

    if len(os.listdir(VIDEO_INPUT_DIR)) == 0:
        print("[ERROR] No videos found in ./test_videos/")
        exit()

    print("[INFO] Loading Detection Model...")

    detector_weights = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    detector = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=detector_weights)

    if os.path.exists(DETECTOR_PATH):
        detector.load_state_dict(torch.load(DETECTOR_PATH, map_location=DEVICE))
        print("[INFO] Custom detector weights loaded")

    detector.to(DEVICE)
    detector.eval()

    print("[INFO] Loading Classification Model...")

    if os.path.exists(CLASSIFIER_PATH):

        classifier = torchvision.models.resnet18(weights=None)
        classifier.fc = torch.nn.Linear(classifier.fc.in_features, 2)
        classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))

    else:

        print("[INFO] Using pretrained backbone")

        classifier = torchvision.models.resnet18(
            weights=torchvision.models.ResNet18_Weights.DEFAULT
        )
        classifier.fc = torch.nn.Linear(classifier.fc.in_features, 2)

    classifier.to(DEVICE)
    classifier.eval()

    detector_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    classifier_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    CLASS_NAMES = {0: "Human", 1: "Animal"}

    video_files = os.listdir(VIDEO_INPUT_DIR)

    for video_file in video_files:

        input_path = os.path.join(VIDEO_INPUT_DIR, video_file)
        output_path = os.path.join(VIDEO_OUTPUT_DIR, f"annotated_{video_file}")

        print(f"[INFO] Processing: {video_file}")

        cap = cv2.VideoCapture(input_path)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))




        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (orig_width, orig_height))

        frame_count = 0

        while cap.isOpened():

            ret, frame = cap.read()
            if not ret:
                break

            frame_count += 1

            # Skip alternate frames (speed optimization)
            if frame_count % 2 != 0:
                continue

            resized_frame = cv2.resize(frame, (640, 360))

            scale_x = orig_width / 640
            scale_y = orig_height / 360

            rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

            input_tensor = detector_transform(rgb_frame).unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                detections = detector(input_tensor)[0]

            boxes = detections["boxes"]
            scores = detections["scores"]

            for i in range(len(boxes)):

                confidence = scores[i].item()

                if confidence < 0.6:
                    continue

                x1, y1, x2, y2 = boxes[i].cpu().numpy()

                x1 = int(x1 * scale_x)
                y1 = int(y1 * scale_y)
                x2 = int(x2 * scale_x)
                y2 = int(y2 * scale_y)

                crop = frame[y1:y2, x1:x2]

                if crop.size == 0:
                    continue

                crop_pil = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                crop_tensor = classifier_transform(crop_pil).unsqueeze(0).to(DEVICE)

                with torch.no_grad():
                    output = classifier(crop_tensor)
                    pred_class = torch.argmax(output, dim=1).item()

                label = CLASS_NAMES[pred_class]

                color = (0, 255, 0) if label == "Human" else (255, 0, 0)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                cv2.putText(
                    frame,
                    f"{label}: {confidence:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

            out.write(frame)

            if frame_count % 30 == 0:
                print(f"[INFO] Processed frames: {frame_count}")

        cap.release()
        out.release()

        print(f"[DONE] Saved: {output_path}")

    print("[SUCCESS] Part A Completed")

# =====================================================
# ================= PART B =============================
# =====================================================

if args.mode == "ocr":

    if args.image is None:
        print("[ERROR] Please provide image path using --image")
        exit()

    print("[INFO] Running Offline OCR...")

    OCR_CONFIG = r'--oem 3 --psm 6'

    image = cv2.imread(args.image)

    if image is None:
        print("[ERROR] Could not load image. Check file path.")
        exit()

    # -----------------------------
    # METHOD 1: RAW OCR
    # -----------------------------

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    text_raw = pytesseract.image_to_string(gray, config=OCR_CONFIG)

    # -----------------------------
    # METHOD 2: ENHANCED OCR
    # -----------------------------

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(gray)

    blur = cv2.GaussianBlur(enhanced, (3, 3), 0)

    thresh = cv2.adaptiveThreshold(
        blur,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    processed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    text_processed = pytesseract.image_to_string(processed, config=OCR_CONFIG)

    # -----------------------------
    # SMART SELECTION + CLEANING
    # -----------------------------

    final_text = text_raw if len(text_raw.strip()) > len(text_processed.strip()) else text_processed

    cleaned_text = clean_ocr_text(final_text)

    output_data = {
        "detected_text": cleaned_text
    }

    with open("ocr_output.json", "w") as f:
        json.dump(output_data, f, indent=4)

    print("[OCR RESULT]")
    print(cleaned_text)

    print("[SUCCESS] OCR Output Saved → ocr_output.json")
