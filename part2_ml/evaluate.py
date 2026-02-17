import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from torch import nn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import cv2
from PIL import Image

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Transform
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Load dataset (only needed for evaluation)
dataset_path = "part2_ml/dataset"
dataset = torchvision.datasets.ImageFolder(dataset_path, transform=transform)
class_names = dataset.classes

# Load model
model = models.resnet18(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, len(class_names))

model.load_state_dict(torch.load("part2_ml/model.pth", map_location=device))
model = model.to(device)
model.eval()


# -----------------------------
# Prediction function (for Part 4)
# -----------------------------
def predict_image(cv2_image):
    model.eval()

    # Convert BGR (OpenCV) to RGB
    image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

    # Convert to PIL
    pil_image = Image.fromarray(image)

    # Apply transform
    input_tensor = transform(pil_image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = torch.max(output, 1)

    return class_names[pred.item()]


# -----------------------------
# Optional: Evaluation (Part 2 requirement)
# -----------------------------
if __name__ == "__main__":
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=class_names))

    print("Confusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))
