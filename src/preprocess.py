import os
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split

# ---------- PATHS (relative to src/) ----------
RAW_DYS = os.path.join("..", "data", "raw", "dyslexic")
RAW_NON = os.path.join("..", "data", "raw", "non_dyslexic")

PROC_DYS = os.path.join("..", "data", "processed", "dyslexic")
PROC_NON = os.path.join("..", "data", "processed", "non_dyslexic")

SPLITS_DIR = os.path.join("..", "data", "splits")


def ensure_dirs():
    """Create required folders if they don't exist."""
    os.makedirs(RAW_DYS, exist_ok=True)
    os.makedirs(RAW_NON, exist_ok=True)
    os.makedirs(PROC_DYS, exist_ok=True)
    os.makedirs(PROC_NON, exist_ok=True)
    os.makedirs(SPLITS_DIR, exist_ok=True)


def preprocess_folder(input_dir, output_dir, label, rows, max_images=200):
    """
    Read images from input_dir, preprocess, save to output_dir,
    and append (path, label) to rows list.
    max_images: limit per class for faster testing.
    """
    count = 0
    for fname in os.listdir(input_dir):
        in_path = os.path.join(input_dir, fname)
        if not os.path.isfile(in_path):
            continue  # skip folders

        img = cv2.imread(in_path)
        if img is None:
            print(f"[WARN] Could not read image: {in_path}, skipping.")
            continue

        img = cv2.resize(img, (224, 224))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(output_dir, fname)
        cv2.imwrite(out_path, img)

        rel_path = out_path.replace("\\", "/")
        rows.append({"image_path": rel_path, "label": label})

        count += 1
        if count % 50 == 0:
            print(f"  processed {count} images from {input_dir}")

        if max_images is not None and count >= max_images:
            print(f"  reached max_images={max_images} for {input_dir}")
            break


def run_preprocessing():
    ensure_dirs()

    rows = []

    print("▶ Preprocessing dyslexic images...")
    preprocess_folder(RAW_DYS, PROC_DYS, 1, rows, max_images=200)

    print("▶ Preprocessing non-dyslexic images...")
    preprocess_folder(RAW_NON, PROC_NON, 0, rows, max_images=200)

    if not rows:
        raise RuntimeError("No images were processed. Check data/raw paths.")

    df = pd.DataFrame(rows)
    print(f"Total samples: {len(df)}")
    print("Label counts:")
    print(df["label"].value_counts())

    # ---------- train / test split ----------
    train_df, test_df = train_test_split(
        df, test_size=0.3, stratify=df["label"], random_state=42
    )

    # ---------- SAVE CSVs ----------
    os.makedirs(SPLITS_DIR, exist_ok=True)

    train_csv_path = os.path.join(SPLITS_DIR, "train.csv")
    test_csv_path = os.path.join(SPLITS_DIR, "test.csv")

    train_df.to_csv(train_csv_path, index=False)
    test_df.to_csv(test_csv_path, index=False)

    print(f"✅ Saved train split to {train_csv_path}")
    print(f"✅ Saved test  split to {test_csv_path}")


if __name__ == "__main__":
    run_preprocessing()
    print("✅ Preprocessing finished.")
