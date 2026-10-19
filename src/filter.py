import cv2
import numpy as np


def to_gray(img):
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def sobel(img):
    gray = to_gray(img)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    return cv2.convertScaleAbs(cv2.magnitude(sobelx, sobely))


def blur_sobel(img):
    return sobel(cv2.GaussianBlur(to_gray(img), (3, 3), 0))


PREPROCESSORS = {
    "gray": to_gray,
    "sobel": blur_sobel,
}


def _sample(value, rng):
    if isinstance(value, (tuple, list)):
        return rng.uniform(*value)
    return value


def apply_uav_effects(uav_img, brightness=20, contrast=1.1, noise_level=15, rng=None):
    rng = np.random.default_rng() if rng is None else rng
    brightness = _sample(brightness, rng)
    contrast = _sample(contrast, rng)
    noise_level = _sample(noise_level, rng)

    adjusted = cv2.convertScaleAbs(uav_img, alpha=contrast, beta=brightness)
    noise = rng.normal(0.0, noise_level, adjusted.shape)
    noisy = np.clip(np.rint(adjusted.astype(np.float32) + noise), 0, 255)
    return noisy.astype(np.uint8)