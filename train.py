import os
import argparse

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from models import build_aod_net, build_attention_unet, build_generator, build_discriminator
from utils import list_pairs, data_generator, IMG_SIZE

DATASET_DIR = "dataset"
SAVE_DIR = "saved_models"
os.makedirs(SAVE_DIR, exist_ok=True)


def get_pairs(split="ITS"):
    hazy_dir = os.path.join(DATASET_DIR, split, "hazy")
    clear_dir = os.path.join(DATASET_DIR, split, "clear")
    pairs = list_pairs(hazy_dir, clear_dir)
    if not pairs:
        raise RuntimeError(
            f"No training pairs found in {hazy_dir} / {clear_dir}. "
            "Download the RESIDE dataset and place ITS (train) and SOTS (test) "
            "images there — see README.md."
        )
    return pairs


def train_supervised(model_name, epochs, batch_size, lr=1e-3):
    builder = {"aod_net": build_aod_net, "attention_unet": build_attention_unet}[model_name]
    model = builder(input_shape=(*IMG_SIZE, 3))

    pairs = get_pairs("ITS")
    steps_per_epoch = max(1, len(pairs) // batch_size)
    gen = data_generator(pairs, batch_size=batch_size)

    lr_schedule = tf.keras.optimizers.schedules.CosineDecay(lr, epochs * steps_per_epoch)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss="mse")

    history = model.fit(gen, steps_per_epoch=steps_per_epoch, epochs=epochs)

    model.save(os.path.join(SAVE_DIR, f"{model_name}.keras"))
    _plot_loss(history.history["loss"], model_name)
    print(f"Saved {model_name} to {SAVE_DIR}/{model_name}.keras")


def train_gan(epochs, batch_size, lr=1e-3, lambda_content=100.0):
    generator = build_generator(input_shape=(*IMG_SIZE, 3))
    discriminator = build_discriminator(input_shape=(*IMG_SIZE, 3))

    g_opt = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.999)
    d_opt = tf.keras.optimizers.Adam(lr, beta_1=0.9, beta_2=0.999)

    bce = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    mse = tf.keras.losses.MeanSquaredError()

    pairs = get_pairs("ITS")
    np.random.shuffle(pairs)
    pairs = pairs[:1000]
    print(f"Using a subset of {len(pairs)} pairs for GAN training (full dataset too slow on this hardware).")
    steps_per_epoch = max(1, len(pairs) // batch_size)
    gen_data = data_generator(pairs, batch_size=batch_size)
    losses = []
    for epoch in range(epochs):
        epoch_g_loss = 0.0
        for step in range(steps_per_epoch):
            hazy, clear = next(gen_data)
            hazy = tf.convert_to_tensor(hazy, dtype=tf.float32)
            clear = tf.convert_to_tensor(clear, dtype=tf.float32)

            with tf.GradientTape() as d_tape:
                fake = generator(hazy, training=True)
                real_pred = discriminator(clear, training=True)
                fake_pred = discriminator(fake, training=True)
                d_loss = bce(tf.ones_like(real_pred), real_pred) + \
                          bce(tf.zeros_like(fake_pred), fake_pred)
            d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
            d_opt.apply_gradients(zip(d_grads, discriminator.trainable_variables))

            with tf.GradientTape() as g_tape:
                fake = generator(hazy, training=True)
                fake_pred = discriminator(fake, training=True)
                adv_loss = bce(tf.ones_like(fake_pred), fake_pred)
                content_loss = mse(clear, fake)
                g_loss = adv_loss + lambda_content * content_loss
            g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
            g_opt.apply_gradients(zip(g_grads, generator.trainable_variables))

            epoch_g_loss += float(content_loss)
            if (step + 1) % 10 == 0:
                print(f"  step {step+1}/{steps_per_epoch}  content_mse={float(content_loss):.4f}")

        avg_loss = epoch_g_loss / steps_per_epoch
        losses.append(avg_loss)
        print(f"[GAN] epoch {epoch+1}/{epochs}  content_mse={avg_loss:.4f}")

    generator.save(os.path.join(SAVE_DIR, "gan_generator.keras"))
    _plot_loss(losses, "gan_generator")
    print(f"Saved GAN generator to {SAVE_DIR}/gan_generator.keras")


def _plot_loss(loss_values, model_name):
    plt.figure()
    plt.plot(loss_values)
    plt.title(f"{model_name} training loss")
    plt.xlabel("epoch")
    plt.ylabel("MSE loss")
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/{model_name}_loss.png")
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["aod_net", "attention_unet", "gan"], required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    if args.model == "gan":
        train_gan(args.epochs, args.batch_size, args.lr)
    else:
        train_supervised(args.model, args.epochs, args.batch_size, args.lr)
