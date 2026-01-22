

# This file handles:

# | Task                    | Why                  |
# | ----------------------- | -------------------- |
# | Draw bounding boxes     | Visualization        |
# | Put labels & confidence | Interpretability     |
# | Read & write videos     | End-to-end pipeline  |
# | FPS measurement         | Performance analysis |



import cv2
import time
import os


# 1. Draw Bounding Boxes

def draw_boxes(frame, boxes, labels, scores):
    for box, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = map(int, box)

        color = (0, 255, 0) if label == "human" else (0, 0, 255)

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        text = f"{label.upper()} : {score:.2f}"
        cv2.putText(
            frame,
            text,
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            color,
            2
        )

    return frame

# Green → Human
# Red → Animal


# 2. FPS Counter

class FPSCounter:
    def __init__(self):
        self.start_time = time.time()
        self.frames = 0

    def update(self):
        self.frames += 1
        elapsed = time.time() - self.start_time
        return self.frames / elapsed if elapsed > 0 else 0



# 3. Video Reader

def open_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise IOError("Cannot open video")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    return cap, width, height, fps



# 4. Video Writer

def create_video_writer(save_path, width, height, fps):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(
        save_path,
        fourcc,
        fps,
        (width, height)
    )

    return writer



