import torch.nn as nn
from torchvision import models

def build_model():
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    
    # Change first layer to accept 1 channel instead of 3
    model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

    # Replace final FC layer for binary classification
    model.fc = nn.Linear(model.fc.in_features, 1)

    return model
