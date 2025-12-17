"""
Combined EDA + Data Preprocessing for SpaceNet-7
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import rasterio
import cv2
import random
from collections import defaultdict
import torch
from torch.utils.data import Dataset, DataLoader


def find_tile_root(base_folder):
    """
    Recursively find the folder that contains tile directories.
    A tile directory contains an 'images_masked' subfolder.
    """
    for root, dirs, files in os.walk(base_folder):
        for d in dirs:
            images_path = os.path.join(root, d, "images_masked")
            if os.path.isdir(images_path):
                return root  # this root contains tile folders
    raise ValueError("No tile folders with images_masked/ found inside dataset!")


# CONFIG
# ============================================================

# Automatically detect correct dataset path
DATASET_DIR = find_tile_root(os.path.join(os.getcwd(), "dataset"))
print("[INFO] Auto-detected tile directory:", DATASET_DIR)
#DATASET_DIR = os.path.join(os.getcwd(), "dataset")  # <-- auto-load from ./dataset
IMG_SIZE = 256
SEQ_LENGTH = 3   # number of past frames for future prediction


# Helper Functions
# ============================================================

def extract_timestamp(filepath):
    """
    Extract timestamp from filename:
    global_monthly_2019_04_mosaic_XXXX.tif → year=2019, month=4
    """
    name = os.path.basename(filepath)
    parts = name.split("_")
    return int(parts[2]), int(parts[3])


def load_tif(path):
    """Load TIFF as CHW float32 normalized + resized."""
    with rasterio.open(path) as src:
        img = src.read()[:3]   # keep only RGB channels
  # CHW format

    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    # Resize each band
    resized = np.zeros((img.shape[0], IMG_SIZE, IMG_SIZE), dtype=np.float32)
    for c in range(img.shape[0]):
        resized[c] = cv2.resize(img[c], (IMG_SIZE, IMG_SIZE))

    return resized


# EDA MODULE
# ============================================================

def eda_tile_distribution(root=DATASET_DIR):
    """
    Count number of frames per tile and visualize.
    """
    counts = {}

    for tile in os.listdir(root):
        tile_path = os.path.join(root, tile, "images_masked")
        if not os.path.isdir(tile_path):
            continue

        frames = glob.glob(os.path.join(tile_path, "*.tif"))
        counts[tile] = len(frames)

    print("\n=== Frames per Tile ===")
    for t, c in counts.items():
        print(f"{t}: {c}")

    # Plot
    plt.figure(figsize=(12, 4))
    plt.bar(counts.keys(), counts.values())
    plt.xticks(rotation=90)
    plt.title("Frames per Tile")
    plt.tight_layout()
    plt.show()


def eda_missing_months(root=DATASET_DIR):
    """
    Identify missing timestamps for each tile.
    """
    print("\n=== Missing / Present Months ===")
    for tile in os.listdir(root):
        tile_path = os.path.join(root, tile, "images_masked")
        if not os.path.isdir(tile_path):
            continue

        files = glob.glob(os.path.join(tile_path, "*.tif"))
        timestamps = sorted(extract_timestamp(f) for f in files)
        print(f"\nTile: {tile}")
        print(sorted(timestamps))


def eda_year_distribution(root=DATASET_DIR):
    """
    Plot frame counts by year.
    """
    year_count = defaultdict(int)

    for tile in os.listdir(root):
        tile_path = os.path.join(root, tile, "images_masked")
        if not os.path.isdir(tile_path):
            continue

        files = glob.glob(os.path.join(tile_path, "*.tif"))
        for f in files:
            y, m = extract_timestamp(f)
            year_count[y] += 1

    print("\n=== Frames by Year ===")
    print(dict(year_count))

    plt.figure(figsize=(6, 4))
    plt.bar(year_count.keys(), year_count.values())
    plt.title("Year Distribution")
    plt.xlabel("Year")
    plt.ylabel("Frames")
    plt.tight_layout()
    plt.show()


def eda_image_statistics(root=DATASET_DIR, samples=5):
    """
    Show min/max/mean/std for random sample frames.
    """
    all_frames = []
    for tile in os.listdir(root):
        tile_path = os.path.join(root, tile, "images_masked")
        if os.path.isdir(tile_path):
            all_frames.extend(glob.glob(os.path.join(tile_path, "*.tif")))

    sample_files = random.sample(all_frames, samples)

    print("\n=== Image Stats (Random Samples) ===")
    for f in sample_files:
        img = load_tif(f)
        print(f"\nFile: {f}")
        print("Shape:", img.shape)
        print(f"Min: {img.min():.4f}, Max: {img.max():.4f}")
        print(f"Mean: {img.mean():.4f}, Std: {img.std():.4f}")


def eda_visualize_random(root=DATASET_DIR, samples=3):
    """
    Show random RGB visualizations.
    """
    all_frames = []
    for tile in os.listdir(root):
        tile_path = os.path.join(root, tile, "images_masked")
        if os.path.isdir(tile_path):
            all_frames.extend(glob.glob(os.path.join(tile_path, "*.tif")))

    sample_files = random.sample(all_frames, samples)

    for f in sample_files:
        img = load_tif(f)
        rgb = np.transpose(img[:3], (1, 2, 0))

        plt.figure(figsize=(5, 5))
        plt.imshow(rgb)
        plt.title(os.path.basename(f))
        plt.axis("off")
        plt.show()


# DATASET CLASS (for U-Net, GAN, Siamese)
# ============================================================

class SpaceNet7Dataset(Dataset):
    """
    Creates sequences of images:
    X = [t1, t2, t3]
    Y = next frame (t4)
    """
    def __init__(self, root=DATASET_DIR, seq_length=SEQ_LENGTH, future_prediction=True):
        self.root = root
        self.seq_len = seq_length
        self.future_prediction = future_prediction

        self.data = []  # list of (sequence_paths, target_path)

        for tile in os.listdir(root):
            tile_path = os.path.join(root, tile, "images_masked")
            if not os.path.isdir(tile_path):
                continue

            frames = sorted(
                glob.glob(os.path.join(tile_path, "*.tif")),
                key=lambda f: extract_timestamp(os.path.basename(f))
            )

            # Build sequences
            for i in range(len(frames) - seq_length):
                seq = frames[i:i + seq_length]
                target = frames[i + seq_length]
                self.data.append((seq, target))

        print(f"[INFO] Loaded {len(self.data)} temporal samples")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_paths, target_path = self.data[idx]

        seq_imgs = [load_tif(p) for p in seq_paths]
        seq_tensor = torch.tensor(np.stack(seq_imgs))  # (T,C,H,W)

        target_img = load_tif(target_path)
        target_tensor = torch.tensor(target_img)  # (C,H,W)

        if self.future_prediction:
            return seq_tensor, target_tensor
        else:
            return seq_tensor



# MAIN: RUN EDA + PREPROCESSING
# ============================================================

if __name__ == "__main__":

    print("\n======== RUNNING EDA ========")
    eda_tile_distribution()
    eda_missing_months()
    eda_year_distribution()
    eda_image_statistics(samples=5)
    eda_visualize_random(samples=3)

    print("\n======== BUILDING DATASET ========")
    dataset = SpaceNet7Dataset()
    loader = DataLoader(dataset, batch_size=2, shuffle=True)

    for x, y in loader:
        print("\nBatch shapes:")
        print("X:", x.shape)   # (B,T,C,H,W)
        print("Y:", y.shape)   # (B,C,H,W)
        break
