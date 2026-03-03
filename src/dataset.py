import os
import cv2
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader


class HandwritingDataset(Dataset):
    def __init__(self, csv_path, transform=None, augment=False):
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.transform = transform
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Get row from CSV
        row = self.df.iloc[idx]

        # Column names must match what we wrote in preprocess.py
        img_path = row["image_path"]
        label = int(row["label"])

        # Make path absolute (relative to this file)
        if not os.path.isabs(img_path):
            base_dir = os.path.dirname(__file__)
            img_path = os.path.normpath(os.path.join(base_dir, img_path))

        # Load grayscale image
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {img_path}")

        # Convert to float tensor in range [0, 1] with shape (1, H, W)
        img = img.astype("float32") / 255.0
        img = torch.from_numpy(img).unsqueeze(0)

        # Apply optional transforms
        if self.transform is not None:
            img = self.transform(img)

        return img, torch.tensor(label, dtype=torch.float32)


def get_loader(csv_path, batch=16, augment=False):
    """Helper used by train.py to create DataLoader."""
    dataset = HandwritingDataset(csv_path, augment=augment)
    loader = DataLoader(dataset, batch_size=batch, shuffle=True)
    return loader
