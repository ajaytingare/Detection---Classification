

# This file defines WHAT the model is.

# dataset_loader.py → how data comes

# architecture.py → how the model thinks

# train_*.py → how it learns

# inference_pipeline.py → how it works in real life



# | Model        | Role                     |
# | ------------ | ------------------------ |
# | Faster R-CNN | Detect living objects    |
# | EfficientNet | Classify Human vs Animal |






import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.detection import fasterrcnn_resnet50_fpn



# 1. Faster R-CNN Detector (Stage-1)

def get_detector(num_classes=2):
    """
    num_classes:
    0 -> background
    1 -> living object (human or animal)
    """

    model = fasterrcnn_resnet50_fpn(pretrained=True)

    in_features = model.roi_heads.box_predictor.cls_score.in_features

    model.roi_heads.box_predictor = models.detection.faster_rcnn.FastRCNNPredictor(
        in_features, num_classes
    )

    return model


# | Decision                | Reason                     |
# | ----------------------- | -------------------------- |
# | Faster R-CNN            | High localization accuracy |
# | Single foreground class | Simplifies detection       |
# | Pretrained backbone     | Faster convergence         |
# | Custom head             | Matches our task           |




# 2. EfficientNet Classifier (Stage-2)

class HumanAnimalClassifier(nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()

        self.backbone = models.efficientnet_b0(pretrained=True)

        in_features = self.backbone.classifier[1].in_features

        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        return self.backbone(x)


# | Feature             | Benefit               |
# | ------------------- | --------------------- |
# | Lightweight         | Faster inference      |
# | Strong features     | High accuracy         |
# | ImageNet pretrained | Better generalization |



# 3. Load Saved Models (Inference Helper)

def load_detector(weight_path, device):
    model = get_detector()
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_classifier(weight_path, device):
    model = HumanAnimalClassifier()
    model.load_state_dict(torch.load(weight_path, map_location=device))
    model.to(device)
    model.eval()
    return model

