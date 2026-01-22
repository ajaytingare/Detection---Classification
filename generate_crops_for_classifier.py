
"""
generate_crops_for_classifier.py

Purpose:
Generate cropped images for classifier training using VOC ground-truth boxes.

Output:
data/processed/
├── human/
└── animal/
"""

import os
import xml.etree.ElementTree as ET
from PIL import Image

# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_DIR = "data/raw/JPEGImages"
ANNOT_DIR = "data/raw/Annotations"
OUTPUT_DIR = "data/processed"

HUMAN_CLASSES = ["person"]
ANIMAL_CLASSES = ["dog", "cat", "cow", "horse", "sheep", "bird"]

MIN_BOX_SIZE = 20  # ignore very small crops


# --------------------------------------------------
# Create output folders
# --------------------------------------------------

human_dir = os.path.join(OUTPUT_DIR, "human")
animal_dir = os.path.join(OUTPUT_DIR, "animal")

os.makedirs(human_dir, exist_ok=True)
os.makedirs(animal_dir, exist_ok=True)


# --------------------------------------------------
# Helper: Parse VOC XML
# --------------------------------------------------

def parse_voc_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    objects = []
    for obj in root.findall("object"):
        name = obj.find("name").text.lower()
        bbox = obj.find("bndbox")

        xmin = int(float(bbox.find("xmin").text))
        ymin = int(float(bbox.find("ymin").text))
        xmax = int(float(bbox.find("xmax").text))
        ymax = int(float(bbox.find("ymax").text))

        objects.append((name, xmin, ymin, xmax, ymax))

    return objects


# --------------------------------------------------
# Main crop generation loop
# --------------------------------------------------

human_count = 0
animal_count = 0

image_files = sorted(os.listdir(IMAGE_DIR))

for img_name in image_files:
    if not img_name.endswith(".jpg"):
        continue

    img_path = os.path.join(IMAGE_DIR, img_name)
    xml_path = os.path.join(ANNOT_DIR, img_name.replace(".jpg", ".xml"))

    if not os.path.exists(xml_path):
        continue

    image = Image.open(img_path).convert("RGB")
    objects = parse_voc_xml(xml_path)

    for name, xmin, ymin, xmax, ymax in objects:
        w = xmax - xmin
        h = ymax - ymin

        if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
            continue

        crop = image.crop((xmin, ymin, xmax, ymax))

        if name in HUMAN_CLASSES:
            save_path = os.path.join(
                human_dir, f"human_{human_count:05d}.jpg"
            )
            crop.save(save_path)
            human_count += 1

        elif name in ANIMAL_CLASSES:
            save_path = os.path.join(
                animal_dir, f"animal_{animal_count:05d}.jpg"
            )
            crop.save(save_path)
            animal_count += 1


print("Crop generation completed!")
print(f"Human crops  : {human_count}")
print(f"Animal crops : {animal_count}")
