import torch
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)
from sklearn.preprocessing import label_binarize
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import os

# =========================
# CONFIG
# =========================
DATA_DIR = "../data/PlantVillage_Split/test"   # test folder
MODEL_PATH = "../models/mobilenetv3_best2.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# =========================
# TRANSFORMS
# =========================
eval_tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# =========================
# LOAD DATA
# =========================
test_data = datasets.ImageFolder(DATA_DIR, transform=eval_tfms)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

class_names = test_data.classes
n_classes = len(class_names)

print("Classes:", n_classes)

# =========================
# LOAD MODEL
# =========================
model = models.mobilenet_v3_large(weights=None)
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features,
    n_classes
)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# =========================
# INFERENCE
# =========================
y_true = []
y_pred = []
y_scores = []

with torch.no_grad():
    for imgs, labels in test_loader:
        imgs = imgs.to(device)

        outputs = model(imgs)
        probs = torch.softmax(outputs, dim=1)

        _, preds = torch.max(outputs, 1)

        y_true.extend(labels.numpy())
        y_pred.extend(preds.cpu().numpy())
        y_scores.extend(probs.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_scores = np.array(y_scores)

# =========================
# CLASSIFICATION REPORT
# =========================
print("\n===== Classification Report =====")
print(classification_report(y_true, y_pred, target_names=class_names))

# =========================
# CONFUSION MATRIX
# =========================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(18,14))
sns.heatmap(
    cm,
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.xticks(rotation=90)
plt.yticks(rotation=0)
plt.tight_layout()
plt.show()

# =========================
# ROC CURVES (MULTI-CLASS)
# =========================
y_true_bin = label_binarize(y_true, classes=list(range(n_classes)))

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# micro average
fpr["micro"], tpr["micro"], _ = roc_curve(
    y_true_bin.ravel(),
    y_scores.ravel()
)
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

plt.figure(figsize=(10,8))

plt.plot(
    fpr["micro"],
    tpr["micro"],
    linewidth=3,
    label=f"Micro Avg ROC (AUC={roc_auc['micro']:.3f})"
)

for i in range(min(n_classes, 6)):   # avoid clutter
    plt.plot(
        fpr[i],
        tpr[i],
        label=f"{class_names[i]} (AUC={roc_auc[i]:.2f})"
    )

plt.plot([0,1],[0,1],'k--')
plt.title("Multi-Class ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid()
plt.show()

# =========================
# PRECISION-RECALL CURVES
# =========================
plt.figure(figsize=(10,8))

for i in range(min(n_classes, 5)):
    precision, recall, _ = precision_recall_curve(
        y_true_bin[:, i],
        y_scores[:, i]
    )
    plt.plot(recall, precision, label=class_names[i])

plt.title("Precision-Recall Curves")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.grid()
plt.show()

# =========================
# PER-CLASS ACCURACY BAR
# =========================
per_class_acc = cm.diagonal() / cm.sum(axis=1)

plt.figure(figsize=(12,6))
plt.bar(class_names, per_class_acc)
plt.xticks(rotation=90)
plt.title("Per-Class Accuracy")
plt.ylabel("Accuracy")
plt.show()

# =========================
# CONFIDENCE HISTOGRAM
# =========================
confidences = y_scores.max(axis=1)

plt.hist(confidences, bins=20)
plt.title("Prediction Confidence Distribution")
plt.xlabel("Confidence")
plt.ylabel("Count")
plt.show()

# =========================
# TOP-3 ACCURACY
# =========================
top3 = 0
for i in range(len(y_scores)):
    if y_true[i] in np.argsort(y_scores[i])[-3:]:
        top3 += 1

print("Top-3 Accuracy:", top3 / len(y_true))
