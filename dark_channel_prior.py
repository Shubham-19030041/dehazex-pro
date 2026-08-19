import cv2
import numpy as np


def _dark_channel(img, patch_size=15):
    """img: float32 [0,1], HxWx3. Returns HxW dark channel."""
    min_channel = np.min(img, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (patch_size, patch_size))
    return cv2.erode(min_channel, kernel)


def _atmospheric_light(img, dark_channel, top_percent=0.001):
    h, w = dark_channel.shape
    num_pixels = h * w
    num_top = max(1, int(num_pixels * top_percent))

    flat_dark = dark_channel.reshape(-1)
    flat_img = img.reshape(-1, 3)

    indices = np.argsort(flat_dark)[-num_top:]
    brightest = flat_img[indices]
    return brightest.max(axis=0)  # shape (3,)


def _transmission_estimate(img, atmospheric_light, patch_size=15, omega=0.95):
    normalized = img / np.maximum(atmospheric_light, 1e-6)
    dark = _dark_channel(normalized, patch_size)
    return 1.0 - omega * dark


def _guided_filter(guide_gray, src, radius=40, eps=1e-3):
    """Edge-aware smoothing of the transmission map, guided by the (gray)
    hazy image, so refined transmission respects real edges instead of
    producing blocky artifacts."""
    guide = guide_gray.astype(np.float32)
    src = src.astype(np.float32)

    mean_guide = cv2.boxFilter(guide, cv2.CV_32F, (radius, radius))
    mean_src = cv2.boxFilter(src, cv2.CV_32F, (radius, radius))
    mean_guide_src = cv2.boxFilter(guide * src, cv2.CV_32F, (radius, radius))
    cov_guide_src = mean_guide_src - mean_guide * mean_src

    mean_guide_sq = cv2.boxFilter(guide * guide, cv2.CV_32F, (radius, radius))
    var_guide = mean_guide_sq - mean_guide * mean_guide

    a = cov_guide_src / (var_guide + eps)
    b = mean_src - a * mean_guide

    mean_a = cv2.boxFilter(a, cv2.CV_32F, (radius, radius))
    mean_b = cv2.boxFilter(b, cv2.CV_32F, (radius, radius))

    return mean_a * guide + mean_b


def dehaze_dcp(img, patch_size=15, omega=0.95, t0=0.1, refine=True):
    """
    img: float32 array in [0,1], shape (H, W, 3) — RGB.
    Returns: float32 array in [0,1], dehazed image.
    """
    img = np.clip(img, 0.0, 1.0).astype(np.float32)

    dark = _dark_channel(img, patch_size)
    A = _atmospheric_light(img, dark)
    t = _transmission_estimate(img, A, patch_size, omega)

    if refine:
        gray = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
        t = _guided_filter(gray, t)

    t = np.clip(t, t0, 1.0)
    t3 = np.repeat(t[:, :, np.newaxis], 3, axis=2)

    recovered = (img - A) / t3 + A
    return np.clip(recovered, 0.0, 1.0).astype(np.float32)