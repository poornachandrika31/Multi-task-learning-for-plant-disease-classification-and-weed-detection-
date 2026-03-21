import torch
from torchvision import transforms, models, datasets
from PIL import Image
import random
import os
import matplotlib.pyplot as plt

MODEL_PATH = "../models/mobilenetv3_best.pth"
DATA_DIR = "../data/PlantVillage_Split/test"

# Load dataset to get class names
dataset = datasets.ImageFolder(DATA_DIR)
class_names = dataset.classes

# Load model
model = models.mobilenet_v3_large(weights=None)
model.classifier[3] = torch.nn.Linear(
    model.classifier[3].in_features, len(class_names)
)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# Transforms
tfms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------- Pick random image --------
class_folder = random.choice(os.listdir(DATA_DIR))
img_name = random.choice(os.listdir(os.path.join(DATA_DIR, class_folder)))
img_path = os.path.join(DATA_DIR, class_folder, img_name)

print("Selected Image:", img_path)

img = Image.open(img_path).convert("RGB")
img_tensor = tfms(img).unsqueeze(0)

# -------- Inference --------
with torch.no_grad():
    output = model(img_tensor)
    probs = torch.softmax(output, dim=1)
    confidence, pred = torch.max(probs, 1)

predicted_class = class_names[pred.item()]
confidence = confidence.item() * 100

# -------- Display Image + Prediction --------
plt.imshow(img)
plt.axis("off")
plt.title(f"Predicted: {predicted_class}\nConfidence: {confidence:.2f}%")
plt.show()
