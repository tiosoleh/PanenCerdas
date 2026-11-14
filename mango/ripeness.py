"""Analisis kematangan mangga berbasis warna (CIE-LAB).

CIE-LAB memisahkan luminansi (L*) dari kromatisitas:
  a* : hijau (-) ↔ merah (+)
  b* : biru (-) ↔ kuning (+)

Mangga mentah: a* rendah (hijau), b* sedang
Mangga matang : a* tinggi (merah/kuning), b* tinggi
Rasio kematangan dihitung dari distribusi a* dan b*.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SIAP_PANEN = "Siap panen"
MATANG = "Matang"
SETENGAH = "Setengah matang"
MENTAH = "Mentah"
UNKNOWN = "Tidak diketahui"

# 4 kelas kematangan sesuai standar agronomis:
#   Unripe (Mentah) → Mature (Setengah matang) → Ripe (Matang) → Harvest-ready (Siap panen)
# Kelas "Siap panen" mempertimbangkan bahwa buah matang + siap dipetik
# berdasarkan rasio warna kulit yang tinggi (a* tinggi = kuning/oranye penuh)

# warna kotak (BGR) per status — palet berbeda
COLORS = {
    SIAP_PANEN: (0, 220, 0),       # hijau terang — prioritas panen
    MATANG: (0, 180, 80),           # hijau tua
    SETENGAH: (0, 200, 220),        # kuning/cyan
    MENTAH: (180, 120, 30),         # biru/hijau tua
    UNKNOWN: (140, 140, 140),
}


@dataclass
class RipenessConfig:
    """Konfigurasi analisis kematangan berbasis LAB."""
    a_star_threshold: float = 8.0       # a* >= ini dianggap matang (merah/kuning)
    b_star_min: float = 15.0            # b* minimum untuk piksel yang valid
    luminance_min: float = 25.0         # L* minimum (buang bayangan gelap)
    luminance_max: float = 95.0         # L* maksimum (buang pantulan terang)
    center_crop: float = 0.18           # buang 18% tiap sisi bbox
    harvest_threshold: float = 0.80     # rasio >= ini -> Siap panen (harvest-ready)
    ripe_threshold: float = 0.60        # rasio >= ini -> Matang (ripe)
    mid_threshold: float = 0.30         # antara mid..ripe -> Setengah matang (mature)


@dataclass
class Ripeness:
    ratio: float   # 0 = full hijau, 1 = full kuning/merah
    label: str


def label_from_ratio(ratio: float, cfg: RipenessConfig) -> str:
    """Klasifikasi 4 level berdasarkan rasio piksel matang.

    Sesuai Section 8 UTS — 4 kelas kematangan:
      Unripe (Mentah)       : ratio < mid_threshold
      Mature (Setengah matang): mid_threshold <= ratio < ripe_threshold
      Ripe (Matang)         : ripe_threshold <= ratio < harvest_threshold
      Harvest-ready (Siap panen): ratio >= harvest_threshold
    """
    if ratio >= cfg.harvest_threshold:
        return SIAP_PANEN
    if ratio >= cfg.ripe_threshold:
        return MATANG
    if ratio >= cfg.mid_threshold:
        return SETENGAH
    return MENTAH


def is_harvest_ready(label: str) -> bool:
    """Cek apakah buah termasuk siap panen (untuk rumus N_panen)."""
    return label == SIAP_PANEN


def analyze(crop_bgr: np.ndarray, cfg: RipenessConfig | None = None,
            mask: np.ndarray | None = None) -> Ripeness:
    """Analisis kematangan menggunakan ruang warna CIE-LAB.

    Berbeda dari HSV: LAB memisahkan sumbu hijau-merah (a*) dan biru-kuning (b*)
    secara perseptual, lebih konsisten di berbagai kondisi pencahayaan.

    mask: opsional, 1 = piksel buah. Jika ada, analisis hanya pada piksel buah."""
    cfg = cfg or RipenessConfig()
    h, w = crop_bgr.shape[:2]
    if h < 8 or w < 8:
        return Ripeness(0.0, UNKNOWN)

    if mask is not None:
        inner = cv2.erode(mask, np.ones((3, 3), np.uint8))
        use = inner if inner.sum() >= 20 else mask
        roi, sel = crop_bgr, use.astype(bool)
    else:
        my, mx = int(h * cfg.center_crop), int(w * cfg.center_crop)
        roi, sel = crop_bgr[my:h - my, mx:w - mx], None

    # Konversi ke CIE-LAB
    lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB).astype(np.float32)
    # OpenCV LAB: L [0,255], a [0,255], b [0,255] — perlu dinormalisasi
    # L: 0-255 → 0-100,  a: 0-255 → -128 to 127,  b: 0-255 → -128 to 127
    L = lab[..., 0] * 100.0 / 255.0
    a_star = lab[..., 1] - 128.0
    b_star = lab[..., 2] - 128.0

    # Filter piksel valid: luminansi wajar (bukan bayangan/pantulan)
    valid = (L >= cfg.luminance_min) & (L <= cfg.luminance_max) & (b_star >= cfg.b_star_min)
    if sel is not None:
        valid &= sel

    # Klasifikasi: a* tinggi = matang (kuning/oranye/merah), a* rendah = mentah (hijau)
    matang_mask = valid & (a_star >= cfg.a_star_threshold)
    mentah_mask = valid & (a_star < cfg.a_star_threshold)

    total = int(matang_mask.sum() + mentah_mask.sum())
    if total < 20:
        return Ripeness(0.0, UNKNOWN)

    ratio = float(matang_mask.sum()) / total
    return Ripeness(ratio, label_from_ratio(ratio, cfg))