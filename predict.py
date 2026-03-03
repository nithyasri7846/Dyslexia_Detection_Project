import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2

from model import build_model

MODEL_PATH = "best_model.pth"

# -----------------------------
# Image Preprocessing Function
# -----------------------------
def preprocess_image(img_path):

    # Read image using OpenCV
    img = cv2.imread(img_path)

    if img is None:
        raise ValueError("Invalid image path or corrupted image")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Auto threshold to handle any background
    _, thresh = cv2.threshold(
        gray, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Resize to model input size
    resized = cv2.resize(thresh, (224, 224))

    # Convert to PIL
    pil_img = Image.fromarray(resized)

    # Transform to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])

    tensor = transform(pil_img).unsqueeze(0)

    return tensor


# -----------------------------
# Prediction Function
# -----------------------------
def predict(img_path):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = build_model()
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    # Preprocess input
    input_tensor = preprocess_image(img_path)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        output = model(input_tensor)

    # ---------------------------------
    # If model has single output neuron
    # ---------------------------------
    if output.shape[1] == 1:

        prob = torch.sigmoid(output).item()

        dys_prob = float(prob)
        non_dys_prob = float(1 - prob)

        label = "Dyslexic" if prob > 0.5 else "Non-dyslexic"

    # ---------------------------------
    # If model has 2 output neurons
    # ---------------------------------
    else:

        probs = F.softmax(output, dim=1)

        non_dys_prob = float(probs[0][0].item())
        dys_prob = float(probs[0][1].item())

        label = "Dyslexic" if dys_prob > non_dys_prob else "Non-dyslexic"

    return label, dys_prob, non_dys_prob
