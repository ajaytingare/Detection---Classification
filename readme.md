
# 🧠 MIPL – Human & Animal Detection System

## 📌 Project Overview

This project implements a two-stage computer vision pipeline to detect and classify
humans and animals in videos.

Instead of using a single heavy model, the system is designed in two clear stages:

1. Stage 1 – Object Detection  
   Detects where a living object (human or animal) is present in the frame.

2. Stage 2 – Classification  
   Takes the detected object region and classifies it as Human or Animal.

This modular design improves accuracy, clarity, and scalability.

---

## 🎯 Motivation Behind This Project

In real-world applications such as:
- Drone surveillance
- Wildlife monitoring
- Smart agriculture
- Security and perimeter monitoring

It is not enough to just detect objects.  
We must understand what kind of object is present.

Single-stage detectors often struggle with fine-grained classification.
To solve this, localization and semantic understanding are separated
into two dedicated models.

---

## 🏗️ High-Level Architecture

Video Frame
   ↓
Faster R-CNN (Object Detection)
   ↓
Bounding Box Cropping
   ↓
EfficientNet Classifier
   ↓
Human / Animal Label
   ↓
Annotated Output Video

Each stage has a clear responsibility, making the system easy to debug,
maintain, and extend.

---

## 📁 Project Structure

MIPL_Human_Animal_Detection/
├── data/
│   ├── raw/                # Original PASCAL VOC dataset
│   └── processed/          # Cropped images for classifier training
│
├── models/
│   ├── detector_v1.pth     # Faster R-CNN weights
│   └── classifier_v1.pth   # EfficientNet weights
│
├── test_videos/            # Input videos
├── outputs/                # Output annotated videos
│
├── src/
│   ├── dataset_loader.py   # Dataset handling and preprocessing
│   ├── architecture.py     # Model architectures
│   └── utils.py            # Drawing, video I/O, helpers
│
├── train_detector.py       # Train Faster R-CNN
├── train_classifier.py     # Train EfficientNet classifier
├── inference_pipeline.py   # End-to-end video inference
├── requirements.txt
└── README.md

---

## 🧩 Core File Explanation

### dataset_loader.py
- Loads PASCAL VOC dataset
- Parses XML annotations
- Prepares data for Faster R-CNN
- Crops detected regions for classifier training

This file acts as the data backbone of the project.

---

### architecture.py
Defines both models:
- Faster R-CNN for object detection
- EfficientNet for human vs animal classification

Both models use pretrained backbones for faster convergence
and better generalization.

---

### utils.py
Contains helper utilities such as:
- Drawing bounding boxes and labels
- Video reading and writing
- FPS calculation

Keeps the main logic clean and readable.

---

### train_detector.py
- Trains Faster R-CNN on PASCAL VOC
- Detects any living object (human or animal)
- Saves trained detector weights

This model focuses only on localization.

---

### train_classifier.py
- Uses cropped object images
- Trains EfficientNet to classify:
  - Human
  - Animal
- Saves classifier weights

Separating classification improves accuracy.

---

### inference_pipeline.py
This is the main demo script.

It:
- Takes an input video
- Runs detection and classification
- Draws labels and confidence scores
- Saves the annotated output video

---

## 📊 Why Two-Stage Design?

- Cleaner separation of tasks
- Better localization accuracy
- Improved classification reliability
- Modular and scalable architecture
- Easier debugging and future upgrades

This approach reflects real-world industrial vision systems.

---

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- OpenCV
- NumPy
- PASCAL VOC Dataset

---
```
## ▶️ How to Run

1. Install dependencies
   pip install -r requirements.txt

2. Folder Structure
   python src/setup.py   

3. Train detector
   python train_detector.py

4. Detector Results
   python inference_only_detector.py
   
5. Train classifier
   python train_classifier.py

6. Classifier Results
   python inference_only_classifier.py

7. Run inference
   python inference_pipeline.py

Output video will be saved in the outputs/ directory.
```
---

## 🚀 Applications

- Drone-based monitoring
- Wildlife detection
- Smart farming
- Surveillance systems
- Academic and research projects

---

## 🧠 Key Learning Outcomes

- Two-stage ML pipeline design
- Object detection using Faster R-CNN
- Image classification using EfficientNet
- Video-based inference
- Clean and scalable project structure

---

## 🧑‍💻 Author

Ajay Tingare  
AI / ML Developer

This project was designed and implemented as a learning-focused
and application-oriented computer vision system.
