import os
import torch
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from dataset import get_loader
from model import build_model

def evaluate_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model
    model = build_model()
    model.load_state_dict(torch.load("../models/best_model.pth", map_location=device))
    model.to(device)
    model.eval()

    # Load test data
    loader = get_loader("../data/splits/test.csv", batch=16)

    preds, trues = [], []

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device).unsqueeze(1)
            out = model(imgs)
            probs = torch.sigmoid(out)
            pred = (probs > 0.5).float()

            preds.extend(pred.cpu().numpy().flatten())
            trues.extend(labels.cpu().numpy().flatten())

    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds)
    cm = confusion_matrix(trues, preds)

    print("Accuracy:", acc)
    print("F1 Score:", f1)
    print("Confusion Matrix:\n", cm)


if __name__ == "__main__":
    evaluate_model()
