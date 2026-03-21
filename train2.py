import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

# ====================
# CONFIG
# ====================
DATA_DIR = "../data/PlantVillage_Split"   # must contain train/, val/, test/
SAVE_PATH = "../models/mobilenetv3_best2.pth"

BATCH_SIZE = 32
EPOCHS = 15
LR = 0.0005

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# ====================
# TRANSFORMS
# ====================
train_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(20),
    transforms.ToTensor(),
])

eval_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# ====================
# LOAD DATASETS
# ====================
train_data = datasets.ImageFolder(os.path.join(DATA_DIR, "train"), transform=train_tfms)
val_data   = datasets.ImageFolder(os.path.join(DATA_DIR, "val"), transform=eval_tfms)
test_data  = datasets.ImageFolder(os.path.join(DATA_DIR, "test"), transform=eval_tfms)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(train_data.classes)
print("Number of classes:", num_classes)

# ====================
# MODEL
# ====================
model = models.mobilenet_v3_large(weights="IMAGENET1K_V1")
model.classifier[3] = nn.Linear(
    model.classifier[3].in_features, num_classes
)
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

best_val_acc = 0.0

# ====================
# TRAINING LOOP
# ====================
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, labels = imgs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    print(f"Epoch {epoch+1} | Train Loss: {running_loss:.3f} | Train Acc: {train_acc:.3f}")

    # ====================
    # VALIDATION
    # ====================
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    val_acc = correct / total
    print(f"Validation Accuracy: {val_acc:.3f}")

    # ====================
    # SAVE BEST MODEL
    # ====================
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print("✔ Best model saved")

print("\nTraining complete.")
print("Best Validation Accuracy:", best_val_acc)

# ====================
# FINAL TEST EVALUATION (RUN ONCE)
# ====================
print("\nEvaluating on TEST set...")

model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
model.eval()

correct = 0
total = 0

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs, labels = imgs.to(device), labels.to(device)
        outputs = model(imgs)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

test_acc = correct / total
print("Final Test Accuracy:", test_acc)
