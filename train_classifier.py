"""
train_classifier.py

Stage-2: Human vs Animal Classification

- Trains an image classifier on cropped images
- Logs loss and accuracy using Weights & Biases (wandb)
- Saves trained classifier weights

Expected dataset structure:
data/processed/
├── human/
└── animal/
"""

import os
import torch
import wandb
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

from src.architecture import load_classifier
from src.dataset_loader import get_classifier_transforms
from src.architecture import HumanAnimalClassifier


# --------------------------------------------------
# 1. Configuration
# --------------------------------------------------

DATA_DIR = "data/processed"
MODEL_SAVE_PATH = "models/classifier_v1.pth"

NUM_CLASSES = 2
BATCH_SIZE = 16
NUM_EPOCHS = 4
LEARNING_RATE = 1e-4
TRAIN_SPLIT = 0.8   # 80% train, 20% validation


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
    name="classifier-stage2"
)


# --------------------------------------------------
# 4. Dataset & DataLoaders
# --------------------------------------------------

transform = get_classifier_transforms()

full_dataset = ImageFolder(
    root=DATA_DIR,
    transform=transform
)

num_train = int(len(full_dataset) * TRAIN_SPLIT)
num_val = len(full_dataset) - num_train

train_dataset, val_dataset = random_split(
    full_dataset,
    [num_train, num_val]
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)

print(f"Training samples   : {len(train_dataset)}")
print(f"Validation samples : {len(val_dataset)}")


# --------------------------------------------------
# 5. Model, Loss, Optimizer
# --------------------------------------------------

model = HumanAnimalClassifier(num_classes=NUM_CLASSES)
model.to(device)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# --------------------------------------------------
# 6. Training Loop
# --------------------------------------------------

for epoch in range(NUM_EPOCHS):

    # -------------------------
    # TRAIN
    # -------------------------
    model.train()
    train_loss = 0.0
    train_correct = 0
    train_total = 0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()

        _, preds = torch.max(outputs, 1)
        train_correct += (preds == labels).sum().item()
        train_total += labels.size(0)

    train_loss /= len(train_loader)
    train_accuracy = train_correct / train_total


    # -------------------------
    # VALIDATION
    # -------------------------
    model.eval()
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            val_correct += (preds == labels).sum().item()
            val_total += labels.size(0)

    val_accuracy = val_correct / val_total


    # -------------------------
    # LOGGING
    # -------------------------
    print(
        f"Epoch [{epoch+1}/{NUM_EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Acc: {val_accuracy:.4f}"
    )

    wandb.log({
        "epoch": epoch + 1,
        "classifier_train_loss": train_loss,
        "classifier_train_accuracy": train_accuracy,
        "classifier_val_accuracy": val_accuracy
    })


# --------------------------------------------------
# 7. Save Model
# --------------------------------------------------

os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)

print("Classifier model saved successfully!")
