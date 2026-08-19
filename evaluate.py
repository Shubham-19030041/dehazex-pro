"""
DeHazeX Pro - Evaluation / Comparison Script
---------------------------------------------
Runs all 3 trained models on the SOTS test set, computes PSNR / SSIM /
Entropy / Edge / Time for each, and writes a comparison table (CSV +
Excel) plus a bar chart, exactly like the "Compare" tab in the app.

Usage:
    python evaluate.py
"""

import os
import time

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from utils import list_pairs, load_image, compute_all_metrics
# Importing models.py registers the custom AODFormula layer with Keras —
# required BEFORE load_model() is called (same fix as in web_app.py).
import models  # noqa: F401

DATASET_DIR = "dataset"
SAVE_DIR = "saved_models"
OUT_DIR = "outputs/compare"
os.makedirs(OUT_DIR, exist_ok=True)

MODEL_FILES = {
    "AOD-Net": "aod_net.keras",
    "Attention-UNet": "attention_unet.keras",
    "GAN": "gan_generator.keras",
}


def evaluate_model(model, pairs, max_images=100):
    results = []
    for hazy_path, clear_path in pairs[:max_images]:
        hazy = load_image(hazy_path)
        clear = load_image(clear_path)

        start = time.time()
        dehazed = model.predict(hazy[None, ...], verbose=0)[0]
        elapsed_ms = (time.time() - start) * 1000

        metrics = compute_all_metrics(dehazed, clear, elapsed_ms)
        results.append(metrics)

    df = pd.DataFrame(results)
    return df.mean(numeric_only=True)


def main():
    pairs = list_pairs(os.path.join(DATASET_DIR, "SOTS", "hazy"),
                        os.path.join(DATASET_DIR, "SOTS", "clear"))
    if not pairs:
        raise RuntimeError("No SOTS test pairs found. See README.md for dataset setup.")

    rows = []
    for display_name, filename in MODEL_FILES.items():
        path = os.path.join(SAVE_DIR, filename)
        if not os.path.exists(path):
            print(f"Skipping {display_name}: {path} not found (train it first).")
            continue
        model = tf.keras.models.load_model(path, compile=False, safe_mode=False)
        avg = evaluate_model(model, pairs)
        row = {"Model": display_name}
        row.update(avg.to_dict())
        rows.append(row)
        print(f"{display_name}: {row}")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(OUT_DIR, "comparison.csv")
    xlsx_path = os.path.join(OUT_DIR, "comparison.xlsx")
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    print(f"Saved {csv_path} and {xlsx_path}")

    # Bar chart for PSNR / SSIM
    if not df.empty:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].bar(df["Model"], df["psnr"], color="#4f9dde")
        axes[0].set_title("PSNR (dB)")
        axes[1].bar(df["Model"], df["ssim"], color="#5fd68a")
        axes[1].set_title("SSIM")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, "comparison_bar_chart.png"))
        print(f"Saved chart to {OUT_DIR}/comparison_bar_chart.png")


if __name__ == "__main__":
    main()
