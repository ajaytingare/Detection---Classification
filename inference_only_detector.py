
"""
inference_detector_only.py

Purpose:
Evaluate ONLY Stage-1 (Detector) without classifier.

What this file does:
1. Reads input video
2. Runs Faster R-CNN detector
3. Draws bounding boxes with confidence
4. Saves annotated output video

This is used to evaluate localization quality independently.
"""

import os
import cv2
import torch
from torchvision import transforms

from src.architecture import load_detector
from src.utils import open_video, create_video_writer


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

VIDEO_PATH = "test_videos/billli.mp4"
OUTPUT_PATH = "outputs/billli_detector_only.mp4"

DETECTOR_WEIGHTS = "models/detector_v1.pth"

CONF_THRESHOLD = 0.7


# --------------------------------------------------
# 2. Device Setup
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------
# 3. Load Detector Model
# --------------------------------------------------

detector = load_detector(DETECTOR_WEIGHTS, device)


# --------------------------------------------------
# 4. Open Video
# --------------------------------------------------

cap, width, height, fps = open_video(VIDEO_PATH)
writer = create_video_writer(OUTPUT_PATH, width, height, fps)


# --------------------------------------------------
# 5. Inference Loop (Detector Only)
# --------------------------------------------------

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to tensor
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img_tensor = transforms.ToTensor()(rgb).to(device)

    # Run detector
    with torch.no_grad():
        detections = detector([img_tensor])[0]

    boxes = detections["boxes"].cpu().numpy()
    scores = detections["scores"].cpu().numpy()

    # Draw detections
    for box, score in zip(boxes, scores):
        if score < CONF_THRESHOLD:
            continue

        x1, y1, x2, y2 = map(int, box)

        # Draw bounding box
        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        label = f"OBJECT : {score:.2f}"
        cv2.putText(
            frame,
            label,
            (x1, max(y1 - 10, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

    writer.write(frame)


# --------------------------------------------------
# 6. Cleanup
# --------------------------------------------------

cap.release()
writer.release()
cv2.destroyAllWindows()

print("Detector-only inference completed. Output saved!")
