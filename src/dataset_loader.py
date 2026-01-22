
import os
import cv2
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


# 1. VOC Class Mapping

VOC_CLASSES = [
    "person",
    "dog", "cat", "cow", "horse", "sheep", "bird"
]

# We ignore all other VOC classes


# 2. Parse XML Annotation

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    boxes = []
    labels = []

    for obj in root.findall("object"):
        name = obj.find("name").text

        if name not in VOC_CLASSES:
            continue

        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(1)  # 1 = living object

    return boxes, labels


# | Element     | Meaning                      |
# | ----------- | ---------------------------- |
# | XML parsing | VOC annotations              |
# | boxes       | Bounding boxes               |
# | labels      | Binary class (living object) |
#Here, we only care about living objects.



# 3. VOC Detection Dataset Class

from torchvision import transforms

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, annot_dir):
        self.image_dir = image_dir
        self.annot_dir = annot_dir
        self.images = sorted(os.listdir(image_dir))[:300]

        self.transform = transforms.ToTensor()

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        xml_path = os.path.join(
            self.annot_dir, img_name.replace(".jpg", ".xml")
        )

        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)   # ✅ CONVERT TO TENSOR

        boxes = []
        labels = []

        tree = ET.parse(xml_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            bbox = obj.find("bndbox")

            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(1)  # object

        if len(boxes) == 0:
            return None

        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "labels": torch.tensor(labels, dtype=torch.int64)
        }

        return image, target

    def __len__(self):
        return len(self.images)


# this format is taken because of detectron2 requirements



# 4. Crop Objects for Classifier Dataset

def crop_and_save(image_path, boxes, label, save_dir, img_id):
    img = cv2.imread(image_path)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        crop = img[y1:y2, x1:x2]

        if crop.size == 0:
            continue

        class_dir = "human" if label == "person" else "animal"
        os.makedirs(os.path.join(save_dir, class_dir), exist_ok=True)

        save_path = os.path.join(
            save_dir, class_dir, f"{img_id}_{i}.jpg"
        )

        cv2.imwrite(save_path, crop)

# bridge between stage-1 and stage-2
# Detector → crops → classifier



# 5. Classifier Transforms

def get_classifier_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])




# summary

#  VOC parsing
#  Detection dataset
#  Classifier dataset generation
#  Clean separation of stages