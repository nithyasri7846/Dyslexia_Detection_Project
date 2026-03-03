import os
import torch
from torch import nn, optim
from dataset import get_loader
from model import build_model


def train_model():
    print("▶ Creating data loaders...")
    train_loader = get_loader("../data/splits/train.csv", batch=16, augment=True)
    val_loader = get_loader("../data/splits/test.csv", batch=16)
    print(f"  Train batches: {len(train_loader)}")
    print(f"  Val batches:   {len(val_loader)}")

    print("▶ Building model...")
    model = build_model()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device}")
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    # make sure models directory exists
    models_dir = os.path.join("..", "models")
    os.makedirs(models_dir, exist_ok=True)

    best_val_loss = float("inf")
    num_epochs = 3   # keep small for now so you can see it finish quickly

    for epoch in range(num_epochs):
        print(f"\n===== Epoch {epoch + 1}/{num_epochs} =====")
        model.train()
        total_loss = 0.0

        for batch_idx, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device).unsqueeze(1)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            # print every few batches
            if (batch_idx + 1) % 5 == 0 or (batch_idx + 1) == len(train_loader):
                avg_loss = total_loss / (batch_idx + 1)
                print(
                    f"  [Train] Batch {batch_idx + 1}/{len(train_loader)} "
                    f"- avg loss: {avg_loss:.4f}"
                )

        # VALIDATION
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_idx, (imgs, labels) in enumerate(val_loader):
                imgs = imgs.to(device)
                labels = labels.to(device).unsqueeze(1)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        print(
            f"Epoch {epoch + 1} summary: "
            f"Train Loss={total_loss:.3f}, Val Loss={val_loss:.3f}"
        )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join(models_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✔ Saved best model to {save_path}")


if __name__ == "__main__":
    train_model()
