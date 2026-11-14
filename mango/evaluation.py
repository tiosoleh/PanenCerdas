"""Evaluasi akurasi sistem deteksi dan penghitungan mangga.

Sesuai Section 9 UTS — Evaluasi Akurasi Sistem:
  - Error penghitungan: Error(%) = |Np - Na| / Na × 100%
  - Estimasi siap panen: N_panen = Σ I(Mj = H)
  - Metrik tambahan: Precision, Recall, F1-score, MAE, RMSE, MAPE
  - Unique counting formula: N = |∪ ID(Di)|

Dimana:
  Np = jumlah prediksi (dari sistem)
  Na = jumlah aktual (dari penghitungan manual)
  Mj = kelas kematangan buah ke-j
  H  = kelas siap panen (harvest-ready)
  I  = fungsi indikator (1 jika Mj = H, 0 selainnya)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import ripeness as rp


@dataclass
class CountingEvaluation:
    """Hasil evaluasi penghitungan mangga."""
    n_predicted: int = 0           # Np — jumlah prediksi sistem
    n_actual: int = 0              # Na — jumlah aktual (manual)
    n_total_detections: int = 0    # total deteksi di semua frame/foto
    n_after_duplicate_removal: int = 0  # setelah duplicate removal
    n_harvest_ready: int = 0       # N_panen — buah siap panen

    # per-kelas
    count_per_class: dict[str, int] = field(default_factory=dict)

    @property
    def counting_error_pct(self) -> float | None:
        """Error(%) = |Np - Na| / Na × 100%
        Section 9 UTS — rumus utama evaluasi."""
        if self.n_actual <= 0:
            return None
        return abs(self.n_predicted - self.n_actual) / self.n_actual * 100.0

    @property
    def accuracy_pct(self) -> float | None:
        """Akurasi = 100% - Error(%)"""
        err = self.counting_error_pct
        if err is None:
            return None
        return max(0.0, 100.0 - err)

    @property
    def harvest_ratio(self) -> float:
        """Rasio buah siap panen terhadap total unik."""
        if self.n_predicted <= 0:
            return 0.0
        return self.n_harvest_ready / self.n_predicted

    @property
    def duplicate_removal_rate(self) -> float:
        """Berapa persen duplikat yang berhasil dihilangkan."""
        if self.n_total_detections <= 0:
            return 0.0
        removed = self.n_total_detections - self.n_after_duplicate_removal
        return removed / self.n_total_detections * 100.0


def compute_harvest_estimation(labels: list[str]) -> int:
    """Hitung N_panen = Σ I(Mj = H)

    Fungsi indikator I: bernilai 1 jika kelas kematangan = Siap panen,
    0 untuk kelas lainnya.

    Args:
        labels: list label kematangan setiap buah unik
    Returns:
        N_panen — jumlah buah siap panen
    """
    return sum(1 for label in labels if rp.is_harvest_ready(label))


def compute_counting_error(n_predicted: int, n_actual: int) -> dict:
    """Hitung error penghitungan sesuai Section 9 UTS.

    Error(%) = |Np - Na| / Na × 100%

    Returns:
        dict dengan error_pct, accuracy_pct, n_predicted, n_actual
    """
    if n_actual <= 0:
        return {
            "error_pct": None,
            "accuracy_pct": None,
            "n_predicted": n_predicted,
            "n_actual": n_actual,
            "status": "Masukkan jumlah aktual untuk menghitung error."
        }

    error_pct = abs(n_predicted - n_actual) / n_actual * 100.0
    accuracy_pct = max(0.0, 100.0 - error_pct)
    diff = n_predicted - n_actual

    if diff > 0:
        status = f"Over-count: sistem mendeteksi {diff} buah lebih banyak."
    elif diff < 0:
        status = f"Under-count: sistem melewatkan {abs(diff)} buah."
    else:
        status = "Penghitungan sempurna — sesuai jumlah aktual."

    return {
        "error_pct": round(error_pct, 2),
        "accuracy_pct": round(accuracy_pct, 2),
        "n_predicted": n_predicted,
        "n_actual": n_actual,
        "diff": diff,
        "status": status,
    }


def compute_detection_metrics(tp: int, fp: int, fn: int) -> dict:
    """Hitung metrik deteksi: Precision, Recall, F1-score.

    Precision = TP / (TP + FP) — ketepatan deteksi
    Recall    = TP / (TP + FN) — kemampuan menemukan buah
    F1-score  = 2 × (P × R) / (P + R)

    Args:
        tp: True Positive — buah terdeteksi benar
        fp: False Positive — deteksi salah (bukan buah)
        fn: False Negative — buah tidak terdeteksi
    """
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "tp": tp, "fp": fp, "fn": fn,
    }


def compute_error_metrics(predictions: list[int], actuals: list[int]) -> dict:
    """Hitung MAE, RMSE, MAPE untuk beberapa pohon/sampel.

    MAE  = (1/n) Σ|Pi - Ai|           — rata-rata kesalahan
    RMSE = √((1/n) Σ(Pi - Ai)²)       — besarnya kesalahan
    MAPE = (100/n) Σ|Pi - Ai|/Ai      — kesalahan relatif (%)

    Args:
        predictions: list jumlah prediksi per pohon
        actuals: list jumlah aktual per pohon
    """
    if len(predictions) != len(actuals) or len(predictions) == 0:
        return {"mae": None, "rmse": None, "mape": None}

    preds = np.array(predictions, dtype=np.float64)
    acts = np.array(actuals, dtype=np.float64)
    n = len(preds)

    errors = np.abs(preds - acts)
    mae = float(np.mean(errors))
    rmse = float(np.sqrt(np.mean((preds - acts) ** 2)))

    # MAPE — hindari pembagian nol
    valid = acts > 0
    if valid.sum() > 0:
        mape = float(np.mean(errors[valid] / acts[valid]) * 100)
    else:
        mape = None

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape": round(mape, 2) if mape is not None else None,
        "n_samples": n,
    }


def build_evaluation(n_predicted: int, n_actual: int,
                     n_total_detections: int,
                     labels: list[str]) -> CountingEvaluation:
    """Bangun objek evaluasi lengkap dari hasil deteksi.

    Args:
        n_predicted: jumlah buah unik (setelah duplicate removal)
        n_actual: jumlah aktual dari penghitungan manual (0 jika tidak diisi)
        n_total_detections: total bounding box di semua frame
        labels: label kematangan tiap buah unik
    """
    from collections import Counter
    count_per_class = dict(Counter(labels))
    n_harvest = compute_harvest_estimation(labels)

    return CountingEvaluation(
        n_predicted=n_predicted,
        n_actual=n_actual,
        n_total_detections=n_total_detections,
        n_after_duplicate_removal=n_predicted,
        n_harvest_ready=n_harvest,
        count_per_class=count_per_class,
    )
