import cv2
import os
import numpy as np
import torch # We need torch for the YOLOv5 model

# Assume these helper modules are in your project directory
from data_loader import load_and_preprocess 
from feature_extractor import get_model, extract_features
from similarity import compute_cosine_similarity

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


# --- Step 1: Load the Re-ID Feature Extractor and Gallery ---
gallery_path = "gallery"
gallery_feats = []
labels = []

print("Loading Re-ID model and gallery images...")
reid_model = get_model(device) # Your feature extraction model

# Process each image in the gallery folder
for img_name in os.listdir(gallery_path):
    if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
        path = os.path.join(gallery_path, img_name)
        label = img_name.split('_')[0] 
        image = load_and_preprocess(path)
        feature = extract_features(reid_model, image, device=device)
        gallery_feats.append(feature[0])
        labels.append(label)

print(f"✅ Loaded {len(labels)} gallery images.")


# --- Step 2: Load the YOLOv5 Person Detector ---
print("Loading YOLOv5 person detector...")
# 'yolov5s' is small and fast. 's' stands for small.
# We set detector.classes = [0] to ONLY detect people (class 0 in the COCO dataset).
detector = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
detector.classes = [0] 
detector.to(device)
print("✅ Detector loaded.")


# --- Step 3: Start Webcam and Real-Time Processing ---
cap = cv2.VideoCapture(0)
print("✅ Starting real-time multi-person Re-ID. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5 expects images in RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Get detections from YOLOv5
    results = detector(frame_rgb)
    # The results.xyxy[0] tensor contains all detections: [x1, y1, x2, y2, confidence, class]
    detections = results.xyxy[0]

    # Loop through each detected person
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box) # Get bounding box coordinates

        # Crop the detected person from the frame
        detected_person_roi = frame[y1:y2, x1:x2]
        
        if detected_person_roi.size == 0:
            continue

        try:
            # Save, load, and preprocess the crop for the Re-ID model
            cv2.imwrite("temp_crop.jpg", detected_person_roi)
            query_image = load_and_preprocess("temp_crop.jpg")
            query_feat = extract_features(reid_model, query_image, device=device)

            # --- Compare query features against the gallery ---
            similarities = compute_cosine_similarity([query_feat[0]], gallery_feats)[0]
            top_index = np.argmax(similarities)
            top_score = similarities[top_index]

            # --- Visualize the result with colored boxes ---
            if top_score > 0.8:
                label = f"{labels[top_index]} ({top_score:.2f})"
                color = (0, 255, 0)  # Green for known
            else:
                label = "Unknown"
                color = (0, 0, 255)  # Red for unknown

            # Draw the bounding box and label for THIS person
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        except Exception as e:
            # Draw a simple red box if an error occurs during Re-ID for one person
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)


    # Display the final frame with all detections
    cv2.imshow("Real-Time Multi-Person Re-Identification", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
if os.path.exists("temp_crop.jpg"):
    os.remove("temp_crop.jpg")