"""
DeHazeX Pro - Utilities
-----------------------
Data loading, preprocessing, and the metrics used everywhere in the app:
PSNR, SSIM, Entropy, Edge Preservation Ratio, Haze/Weather detection.
"""

import os
import glob
import time

import cv2
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from skimage.measure import shannon_entropy


IMG_SIZE = (256, 256)


# ---------------------------------------------------------------------------
# Image I/O
# ---------------------------------------------------------------------------
def load_image(path, size=IMG_SIZE):
    """Load an image, resize, return float32 array in [0,1], shape (H,W,3)."""
    img = Image.open(path).convert("RGB").resize(size)
    return np.asarray(img, dtype=np.float32) / 255.0


def to_uint8(img):
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)


def save_image(img_float, path):
    Image.fromarray(to_uint8(img_float)).save(path)


# ---------------------------------------------------------------------------
# Dataset loading (RESIDE-style: dataset/ITS/hazy, dataset/ITS/clear)
# ---------------------------------------------------------------------------
def list_pairs(hazy_dir, clear_dir):
    """
    Matches hazy images to their clear/ground-truth counterpart, handling
    two common RESIDE-style naming conventions:

    1. Exact same filename in both folders (e.g. SOTS-style):
         hazy/0001_0.8_0.2.jpg  <->  clear/0001_0.8_0.2.jpg

    2. Hazy filename's numeric prefix matches a shorter clear filename
       (e.g. ITS-style):
         hazy/1_1_0.90179.png  <->  clear/1.png
    """
    hazy_files = sorted(glob.glob(os.path.join(hazy_dir, "*")))
    pairs = []
    for hf in hazy_files:
        base = os.path.basename(hf)

        # Try 1: exact same filename in the clear folder
        exact_match = os.path.join(clear_dir, base)
        if os.path.exists(exact_match):
            pairs.append((hf, exact_match))
            continue

        # Try 2: numeric-prefix match (e.g. "1_1_0.9.png" -> "1.png")
        prefix = base.split("_")[0].split(".")[0]
        matches = glob.glob(os.path.join(clear_dir, prefix + ".*"))
        if matches:
            pairs.append((hf, matches[0]))

    return pairs

def data_generator(pairs, batch_size=4, size=IMG_SIZE, shuffle=True):
    """Simple python generator yielding (hazy_batch, clear_batch) as np arrays."""
    idx = np.arange(len(pairs))
    while True:
        if shuffle:
            np.random.shuffle(idx)
        for start in range(0, len(idx), batch_size):
            batch_idx = idx[start:start + batch_size]
            if len(batch_idx) == 0:
                continue
            hazy_batch, clear_batch = [], []
            for i in batch_idx:
                hf, cf = pairs[i]
                hazy_batch.append(load_image(hf, size))
                clear_batch.append(load_image(cf, size))
            yield np.stack(hazy_batch), np.stack(clear_batch)


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_psnr(dehazed, clear):
    return float(peak_signal_noise_ratio(clear, dehazed, data_range=1.0))


def compute_ssim(dehazed, clear):
    return float(structural_similarity(clear, dehazed, channel_axis=-1, data_range=1.0))


def compute_entropy(img):
    gray = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2GRAY)
    return float(shannon_entropy(gray))


def compute_edge_preservation(dehazed, clear):
    """Ratio of Canny edges in dehazed vs clear ground truth (or vs input if no GT)."""
    g_dehazed = cv2.cvtColor(to_uint8(dehazed), cv2.COLOR_RGB2GRAY)
    g_clear = cv2.cvtColor(to_uint8(clear), cv2.COLOR_RGB2GRAY)
    edges_dehazed = cv2.Canny(g_dehazed, 100, 200)
    edges_clear = cv2.Canny(g_clear, 100, 200)
    denom = np.count_nonzero(edges_clear)
    if denom == 0:
        return 0.0
    return float(np.count_nonzero(edges_dehazed) / denom)


def compute_all_metrics(dehazed, reference, elapsed_ms):
    """reference = clear ground truth OR the original hazy image (self-referential fallback)."""
    return {
        "psnr": round(compute_psnr(dehazed, reference), 2),
        "ssim": round(compute_ssim(dehazed, reference), 4),
        "entropy": round(compute_entropy(dehazed), 4),
        "edge": round(compute_edge_preservation(dehazed, reference), 4),
        "time_ms": round(elapsed_ms, 1),
    }


# ---------------------------------------------------------------------------
# Haze density / weather detection (used by the "Auto Select" + multi-weather features)
# ---------------------------------------------------------------------------
def detect_weather(img):
    """
    Very lightweight heuristic classifier (as described in the project report):
      - low-light : mean intensity below 60
      - rain      : strong vertical-gradient streaks
      - fog       : uniformly low contrast / low std-dev
      - haze      : default fallback
    img: float32 array in [0,1]
    """
    gray = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2GRAY)
    mean_intensity = gray.mean()
    std_intensity = gray.std()

    if mean_intensity < 60:
        return "low-light"

    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    vertical_strength = np.mean(np.abs(grad_y))
    if vertical_strength > 40 and std_intensity > 30:
        return "rain"

    if std_intensity < 35:
        return "fog"

    return "haze"


def haze_density(img):
    """Rough haze-density label used for the 'Auto Select' model recommendation."""
    gray = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2GRAY)
    std_intensity = gray.std()
    if std_intensity > 55:
        return "Light"
    elif std_intensity > 30:
        return "Medium"
    else:
        return "Heavy"


def recommend_model(img):
    """Auto Select logic: heavy haze -> Attention U-Net, light -> AOD-Net, else GAN."""
    density = haze_density(img)
    if density == "Heavy":
        return "attention_unet"
    if density == "Light":
        return "aod_net"
    return "gan_generator"


def preprocess_for_weather(img, weather):
    """Optional pre-processing before feeding to the network."""
    if weather == "low-light":
        gray = cv2.cvtColor(to_uint8(img), cv2.COLOR_RGB2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2RGB)
        return enhanced_rgb.astype(np.float32) / 255.0
    if weather == "rain":
        filtered = cv2.medianBlur(to_uint8(img), 3)
        return filtered.astype(np.float32) / 255.0
    return img


class Timer:
    """Context manager returning elapsed time in milliseconds."""
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.elapsed_ms = (time.time() - self.start) * 1000
