"""Preprocessing citra mangga menggunakan OpenCV.

Sesuai arsitektur sistem (Section 4 UTS):
  - Resize: menyesuaikan resolusi input
  - Deblur: penajaman gambar (Unsharp Mask / Wiener-like)
  - Brightness & Contrast: normalisasi pencahayaan
  - Crop: opsional, potong area of interest
  - CLAHE: pemerataan histogram adaptif untuk kondisi pencahayaan tidak merata

Semua fungsi menerima dan mengembalikan numpy array BGR (format OpenCV).
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class PreprocessConfig:
    """Konfigurasi preprocessing citra."""
    # Resize
    target_width: int = 0          # 0 = tidak resize
    target_height: int = 0

    # Brightness & Contrast
    brightness: int = 0            # -100 .. +100
    contrast: float = 1.0          # 0.5 .. 3.0

    # Deblur (Unsharp Mask)
    sharpen_enabled: bool = False
    sharpen_amount: float = 1.5    # kekuatan penajaman
    sharpen_radius: float = 1.0    # radius Gaussian (sigma)

    # CLAHE — Contrast Limited Adaptive Histogram Equalization
    clahe_enabled: bool = False
    clahe_clip: float = 2.0        # clip limit
    clahe_grid: int = 8            # ukuran grid

    # Denoise
    denoise_enabled: bool = False
    denoise_strength: int = 10     # h parameter untuk fastNlMeansDenoisingColored

    # Crop (persentase dari tepi, 0 = tidak crop)
    crop_pct: float = 0.0          # 0.0 .. 0.3 — potong % dari setiap sisi


def resize_image(img: np.ndarray, width: int = 0, height: int = 0) -> np.ndarray:
    """Resize gambar. Jika salah satu 0, hitung proporsional."""
    if width <= 0 and height <= 0:
        return img
    h, w = img.shape[:2]
    if width > 0 and height > 0:
        return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)
    if width > 0:
        scale = width / w
    else:
        scale = height / h
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)


def adjust_brightness_contrast(img: np.ndarray, brightness: int = 0,
                                contrast: float = 1.0) -> np.ndarray:
    """Atur brightness (-100..+100) dan contrast (0.5..3.0).

    Rumus: output = clip(contrast * input + brightness)
    Secara matematis menggunakan linear transform pada ruang piksel.
    """
    if brightness == 0 and contrast == 1.0:
        return img
    result = img.astype(np.float32)
    result = contrast * result + brightness
    return np.clip(result, 0, 255).astype(np.uint8)


def unsharp_mask(img: np.ndarray, amount: float = 1.5,
                 sigma: float = 1.0) -> np.ndarray:
    """Penajaman gambar menggunakan Unsharp Mask.

    Prinsip: sharp = original + amount * (original - blurred)
    Metode ini umum digunakan untuk deblur citra pertanian
    karena sederhana dan efektif untuk motion blur ringan.
    """
    if amount <= 0:
        return img
    blurred = cv2.GaussianBlur(img, (0, 0), sigma)
    sharpened = cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def apply_clahe(img: np.ndarray, clip_limit: float = 2.0,
                grid_size: int = 8) -> np.ndarray:
    """CLAHE — Contrast Limited Adaptive Histogram Equalization.

    Diterapkan pada channel L dari ruang warna LAB,
    sehingga hanya luminansi yang diratakan tanpa mengubah warna.
    Berguna untuk foto outdoor dengan pencahayaan tidak merata
    (bayangan pohon, sinar matahari langsung).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit,
                            tileGridSize=(grid_size, grid_size))
    l_eq = clahe.apply(l_ch)
    merged = cv2.merge([l_eq, a_ch, b_ch])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)


def denoise(img: np.ndarray, strength: int = 10) -> np.ndarray:
    """Pengurangan noise menggunakan Non-Local Means Denoising.

    Cocok untuk foto smartphone yang sering memiliki noise ISO tinggi,
    terutama pada kondisi pencahayaan rendah (pagi/sore).
    """
    if strength <= 0:
        return img
    return cv2.fastNlMeansDenoisingColored(img, None, strength, strength, 7, 21)


def crop_center(img: np.ndarray, pct: float = 0.1) -> np.ndarray:
    """Potong tepi gambar sebesar pct dari setiap sisi.

    Berguna untuk menghilangkan elemen di tepi frame
    yang bukan area pohon mangga (tangan, tiang, dll).
    """
    if pct <= 0 or pct >= 0.5:
        return img
    h, w = img.shape[:2]
    top = int(h * pct)
    bottom = h - top
    left = int(w * pct)
    right = w - left
    return img[top:bottom, left:right].copy()


def preprocess(img: np.ndarray, cfg: PreprocessConfig | None = None) -> np.ndarray:
    """Pipeline preprocessing lengkap sesuai konfigurasi.

    Urutan pemrosesan:
      1. Resize (jika target_width/height > 0)
      2. Crop (jika crop_pct > 0)
      3. Denoise (jika enabled)
      4. CLAHE (jika enabled)
      5. Brightness & Contrast
      6. Sharpen / Deblur (jika enabled)

    Returns: gambar BGR yang sudah diproses.
    """
    if cfg is None:
        return img

    result = img.copy()

    # 1. Resize
    if cfg.target_width > 0 or cfg.target_height > 0:
        result = resize_image(result, cfg.target_width, cfg.target_height)

    # 2. Crop
    if cfg.crop_pct > 0:
        result = crop_center(result, cfg.crop_pct)

    # 3. Denoise
    if cfg.denoise_enabled:
        result = denoise(result, cfg.denoise_strength)

    # 4. CLAHE
    if cfg.clahe_enabled:
        result = apply_clahe(result, cfg.clahe_clip, cfg.clahe_grid)

    # 5. Brightness & Contrast
    if cfg.brightness != 0 or cfg.contrast != 1.0:
        result = adjust_brightness_contrast(result, cfg.brightness, cfg.contrast)

    # 6. Sharpen
    if cfg.sharpen_enabled:
        result = unsharp_mask(result, cfg.sharpen_amount, cfg.sharpen_radius)

    return result


def get_preprocessing_info(cfg: PreprocessConfig) -> list[str]:
    """Daftar langkah preprocessing yang aktif (untuk ditampilkan di UI)."""
    steps = []
    if cfg.target_width > 0 or cfg.target_height > 0:
        steps.append(f"Resize → {cfg.target_width}×{cfg.target_height}")
    if cfg.crop_pct > 0:
        steps.append(f"Crop {cfg.crop_pct:.0%} tepi")
    if cfg.denoise_enabled:
        steps.append(f"Denoise (h={cfg.denoise_strength})")
    if cfg.clahe_enabled:
        steps.append(f"CLAHE (clip={cfg.clahe_clip}, grid={cfg.clahe_grid})")
    if cfg.brightness != 0 or cfg.contrast != 1.0:
        steps.append(f"Brightness={cfg.brightness:+d}, Contrast={cfg.contrast:.1f}×")
    if cfg.sharpen_enabled:
        steps.append(f"Sharpen (σ={cfg.sharpen_radius}, amt={cfg.sharpen_amount})")
    return steps
