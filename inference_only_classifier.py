
"""
inference_only_classifier.py

Purpose:
Evaluate ONLY Stage-2 (Classifier) without detector.

Input:
- One image OR a folder of cropped images

Output:
- Predicted label (human / animal)
- Confidence score

This script is used to validate classifier performance independently.
"""

import os
import torch
from PIL import Image
from torchvision import transforms

from src.architecture import load_classifier
from src.dataset_loader import get_classifier_transforms


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

# Path can be:
# - single image  -> "test_classifier_images/img1.jpg"
# - folder        -> "test_classifier_images/"
INPUT_PATH = "test_classifier_images"

CLASSIFIER_WEIGHTS = "models/classifier_v1.pth"

CLASS_NAMES = ["animal", "human"]


# --------------------------------------------------
# 2. Device Setup
# --------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# --------------------------------------------------
# 3. Load Classifier
# --------------------------------------------------

classifier = load_classifier(CLASSIFIER_WEIGHTS, device)
classifier.eval()

transform = get_classifier_transforms()


# --------------------------------------------------
# 4. Helper: Run inference on one image
# --------------------------------------------------

def infer_image(image_path):
    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = classifier(tensor)
        prob = torch.softmax(output, dim=1)[0]

    cls_idx = torch.argmax(prob).item()
    confidence = prob[cls_idx].item()

    label = CLASS_NAMES[cls_idx]

    print(
        f"{os.path.basename(image_path):30s} "
        f"→ {label.upper():6s} "
        f"(confidence: {confidence:.3f})"
    )


# --------------------------------------------------
# 5. Inference Logic
# --------------------------------------------------

if os.path.isfile(INPUT_PATH):
    # Single image
    infer_image(INPUT_PATH)

elif os.path.isdir(INPUT_PATH):
    # Folder of images
    images = sorted(os.listdir(INPUT_PATH))

    for img in images:
        if img.lower().endswith((".jpg", ".png", ".jpeg")):
            infer_image(os.path.join(INPUT_PATH, img))
else:
    raise FileNotFoundError(
        f"Input path not found: {INPUT_PATH}"
    )

print("Classifier-only inference completed.")
