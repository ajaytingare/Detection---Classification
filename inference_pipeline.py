


# (End-to-End Video → Detection → Classification → Output)

# Purpose of inference_pipeline.py

# 1. Reads input video

# 2. Runs Faster R-CNN (where is object?)

# 3. Crops detected regions

# 4. Runs EfficientNet (human or animal?)

# 5. Draws labels & confidence

# 6. Saves annotated video


import os
import cv2
import torch
import numpy as np
from torchvision import transforms

from src.architecture import load_detector, load_classifier
from src.utils import draw_boxes, open_video, create_video_writer
from src.dataset_loader import get_classifier_transforms


# 1. Configuration

VIDEO_PATH = "test_videos/pig_human.mp4"
OUTPUT_PATH = "outputs/pig_human_output_annotated.mp4"

DETECTOR_WEIGHTS = "models/detector_v1.pth"
CLASSIFIER_WEIGHTS = "models/classifier_v1.pth"

CONF_THRESHOLD = 0.7



# 2. Device Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# 3. Load Models

detector = load_detector(DETECTOR_WEIGHTS, device)
classifier = load_classifier(CLASSIFIER_WEIGHTS, device)

clf_transform = get_classifier_transforms()


# 4. Open Video

cap, width, height, fps = open_video(VIDEO_PATH)
writer = create_video_writer(OUTPUT_PATH, width, height, fps)


# 5. Inference Loop (CORE LOGIC)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(rgb).to(device)

    with torch.no_grad():
        detections = detector([img_tensor])[0]

    boxes = detections["boxes"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()

    final_boxes = []
    final_labels = []
    final_scores = []

    for box, score in zip(boxes, scores):
        if score < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        crop_pil = transforms.ToPILImage()(crop)
        crop_tensor = clf_transform(crop_pil).unsqueeze(0).to(device)

        with torch.no_grad():
            output = classifier(crop_tensor)
            prob = torch.softmax(output, dim=1)
            cls = torch.argmax(prob).item()
            confidence = prob[0][cls].item()

        label = "human" if cls == 0 else "animal"

        final_boxes.append(box)
        final_labels.append(label)
        final_scores.append(confidence)

    frame = draw_boxes(frame, final_boxes, final_labels, final_scores)
    writer.write(frame)



# 6. Cleanup

cap.release()
writer.release()
cv2.destroyAllWindows()

print("Inference completed. Output saved!")



# What Happens Per Frame (Interview Gold)
"""
Frame
 ↓
Faster R-CNN → Bounding Boxes
 ↓
Crop Object
 ↓
EfficientNet → Human / Animal
 ↓
Draw Label + Confidence
 ↓
Save Frame

"""


