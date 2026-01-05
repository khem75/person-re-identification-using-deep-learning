# video_reid_yolov5.py

import cv2
import os
import torch
import numpy as np
from data_loader import load_and_preprocess
from feature_extractor import get_model, extract_features
from similarity import compute_cosine_similarity
import argparse

# === YOLOv5 Setup ===
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
yolo_model.conf = 0.5
yolo_model.classes = [0]

# === ReID Model Setup ===
device = 'cpu'
reid_model = get_model(device)

# === Load Gallery Embeddings ===
gallery_path = "gallery"
gallery_feats = []
labels = []

print("Loading gallery images...")
for img_name in os.listdir(gallery_path):
    if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(gallery_path, img_name)
        label = os.path.splitext(img_name)[0]
        image = load_and_preprocess(path)
        feature = extract_features(reid_model, image, device=device)
        gallery_feats.append(feature[0])
        labels.append(label)
print(f"✅ Loaded {len(labels)} gallery images.")

# === Set up video input from a file ===
parser = argparse.ArgumentParser(description="Person Re-identification in a video file.")
parser.add_argument("video_path", help="Path to the video file (e.g., 'my_video.mp4')")
args = parser.parse_args()

cap = cv2.VideoCapture(args.video_path)
if not cap.isOpened():
    print(f"Error: Could not open video file at {args.video_path}")
    exit()

print(f"🚀 Running YOLO + ReID on '{args.video_path}'... Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)
    detections = results.xyxy[0]

    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        person_crop = frame[y1:y2, x1:x2]

        if person_crop.shape[0] < 50 or person_crop.shape[1] < 30:
            continue

        cv2.imwrite("temp_person.jpg", person_crop)
        try:
            img = load_and_preprocess("temp_person.jpg")
            query_feat = extract_features(reid_model, img, device=device)
            sims = compute_cosine_similarity([query_feat[0]], gallery_feats)[0]
            top_idx = np.argmax(sims)
            top_score = sims[top_idx]

            # === THRESHOLD ADJUSTED HERE ===
            if top_score > 0.60: # Lowered from 0.80 to 0.60
                label = f"{labels[top_idx]} ({top_score:.2f})"
                print(f"✅[MATCH] Detected: {labels[top_idx]} with confidence {top_score:.2f}")
                color = (0, 255, 0)
            else:
                label = "Unknown"
                print(f"❌[NO MATCH] Person not identified. Highest similarity with {labels[top_idx]}: {top_score:.2f}")
                color = (0, 0, 255)

        except Exception as e:
            label = "Error"
            print(f"[ERROR] Failed to process cropped image: {e}")
            color = (255, 0, 0)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, label, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    cv2.imshow("YOLO + ReID from Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()