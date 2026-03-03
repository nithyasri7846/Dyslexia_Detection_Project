import os
import cv2
import torch
import numpy as np
from PIL import Image as PILImage
from torchvision import transforms
from torchcam.methods import SmoothGradCAMpp

from model import build_model

MODEL_PATH = "best_model.pth"
OUTPUT_DIR = os.path.join("outputs", "gradcam")
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUT_PATH = os.path.join(OUTPUT_DIR, "gradcam_result.jpg")


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def load_model(device):
    model = build_model()
    state_dict = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def run_gradcam(img_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if img_path is None or not os.path.isfile(img_path):
        raise FileNotFoundError("Invalid image path for GradCAM")

    model = load_model(device)
    transform = build_transform()

    pil_img = PILImage.open(img_path).convert("RGB")
    overlay_img = pil_img.resize((224, 224))
    input_tensor = transform(pil_img).unsqueeze(0).to(device)

    cam_extractor = SmoothGradCAMpp(model, target_layer="layer4")

    with torch.enable_grad():
        output = model(input_tensor)

    if output.shape[1] == 1:
        prob = torch.sigmoid(output).item()
        target_idx = 0
    else:
        target_idx = int(output.argmax(dim=1).item())

    activation_map = cam_extractor(class_idx=target_idx, scores=output)[0]

    cam = activation_map.squeeze().detach().cpu().numpy()
    cam = cam - cam.min()
    cam = cam / cam.max()

    cam_resized = cv2.resize((cam * 255).astype("uint8"), (224, 224))
    heatmap = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)

    overlay_np = cv2.cvtColor(np.array(overlay_img), cv2.COLOR_RGB2BGR)
    blended = cv2.addWeighted(overlay_np, 0.6, heatmap, 0.4, 0)

    result_rgb = cv2.cvtColor(blended, cv2.COLOR_BGR2RGB)
    PILImage.fromarray(result_rgb).save(OUT_PATH)

    return OUT_PATH, target_idx, prob if output.shape[1] == 1 else None