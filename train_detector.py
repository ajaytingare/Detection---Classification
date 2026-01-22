"""
train_detector.py

Stage-1: Object Localization (Detector Only)

- Trains Faster R-CNN for "object vs background"
- Logs training loss and mAP@0.5 using wandb
- Saves trained detector weights

"""

import os
import torch
import wandb
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from src.dataset_loader import VOCDataset
from src.architecture import get_detector


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_DIR = "data/raw"
MODEL_SAVE_PATH = "models/detector_v1.pth"

NUM_CLASSES = 2          # background + object
BATCH_SIZE = 1           # safer on CPU
NUM_EPOCHS = 4
LEARNING_RATE = 1e-3
CONF_THRESHOLD = 0.5


# --------------------------------------------------
# 2. Device Setup
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------
# 3. Initialize W&B
# --------------------------------------------------

wandb.init(
    project="human-animal-detection",
    name="detector-stage1"
)


# --------------------------------------------------
# 4. Dataset & DataLoader
# --------------------------------------------------

dataset = VOCDataset(
    image_dir=os.path.join(DATA_DIR, "JPEGImages"),
    annot_dir=os.path.join(DATA_DIR, "Annotations")
)

def collate_fn(batch):
    batch = [b for b in batch if b is not None]
    return tuple(zip(*batch))

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)


# --------------------------------------------------
# 5. Model & Optimizer
# --------------------------------------------------

model = get_detector(num_classes=NUM_CLASSES)
model.to(device)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# 6. IoU & mAP Utilities
# --------------------------------------------------

def compute_iou(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[2], box2[2])
    yB = min(box1[3], box2[3])

    inter = max(0, xB - xA) * max(0, yB - yA)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


def compute_map(pred_boxes, gt_boxes, iou_thresh=0.5):
    matched = set()
    tp, fp = 0, 0

    for pbox in pred_boxes:
        found = False
        for i, gtbox in enumerate(gt_boxes):
            if i in matched:
                continue
            if compute_iou(pbox, gtbox) >= iou_thresh:
                matched.add(i)
                found = True
                break

        if found:
            tp += 1
        else:
            fp += 1

    fn = len(gt_boxes) - len(matched)

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)

    return precision * recall  # simplified AP


# --------------------------------------------------
# 7. Training Loop
# --------------------------------------------------

for epoch in range(NUM_EPOCHS):

    model.train()
    epoch_loss = 0.0
    batch_count = 0

    for batch in tqdm(dataloader):

        if len(batch) == 0:
            continue

        images, targets = batch
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()
        batch_count += 1

    avg_loss = epoch_loss / max(batch_count, 1)


    # --------------------------------------------------
    # 8. mAP Evaluation
    # --------------------------------------------------

    model.eval()
    map_scores = []

    with torch.no_grad():
        for batch in dataloader:
            if len(batch) == 0:
                continue

            images, targets = batch
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            outputs = model(images)

            for out, tgt in zip(outputs, targets):
                scores = out["scores"].cpu().numpy()
                boxes = out["boxes"].cpu().numpy()
                boxes = boxes[scores > CONF_THRESHOLD]

                gt_boxes = tgt["boxes"].cpu().numpy()
                if len(gt_boxes) == 0:
                    continue

                map_scores.append(
                    compute_map(boxes, gt_boxes)
                )

    mean_map = sum(map_scores) / max(len(map_scores), 1)

    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Loss: {avg_loss:.4f} | mAP@0.5: {mean_map:.4f}"
    )

    # --------------------------------------------------
    # 9. W&B Logging
    # --------------------------------------------------

    wandb.log({
        "epoch": epoch + 1,
        "detector_train_loss": avg_loss,
        "detector_mAP@0.5": mean_map
    })


# --------------------------------------------------
# 10. Save Model
# --------------------------------------------------

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("Detector model saved successfully!")
