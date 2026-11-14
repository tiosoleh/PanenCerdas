"""Pipeline: deteksi (YOLO) -> tracking (ByteTrack) -> penggabungan ID berbasis IoU -> kematangan (LAB) -> anotasi."""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO, YOLOE

from . import ripeness as rp

# Kelas buah yang dianggap mangga pada detektor umum (YOLOE prompt-free)
GENERIC_FRUIT = ("mango", "lime", "lemon", "avocado", "pear", "papaya", "peach", "apricot", "orange",
                 "apple", "grapefruit", "citrus fruit", "fruit")


@dataclass
class Fruit:
    box: tuple[int, int, int, int]
    conf: float
    ripeness: rp.Ripeness
    track_id: int | None = None
    cls_name: str = "mango"


def _crop_thumbnail(crop: np.ndarray, target: int = 150) -> np.ndarray:
    h, w = crop.shape[:2]
    scale = target / max(h, w)
    return cv2.resize(crop, (max(1, round(w * scale)), max(1, round(h * scale))),
                      interpolation=cv2.INTER_AREA)


@dataclass
class TrackRecord:
    """Catatan per-ID selama video: akumulasi rasio warna, posisi, dan thumbnail terbaik."""
    ratios: list[float] = field(default_factory=list)
    frame_start: int = 0
    frame_end: int = 0
    bbox_first: tuple | None = None
    bbox_last: tuple | None = None
    best_score: float = 0.0
    thumbnail: np.ndarray | None = None
    peak_confidence: float = 0.0

    # --- pengukuran multi-arah ---
    widths_px: list[int] = field(default_factory=list)    # lebar bbox (piksel) per frame
    heights_px: list[int] = field(default_factory=list)   # tinggi bbox (piksel) per frame
    aspects: list[float] = field(default_factory=list)    # rasio aspek per frame

    @property
    def avg_ratio(self) -> float:
        return float(np.mean(self.ratios)) if self.ratios else 0.0

    def record(self, t: int, box, ratio: float, score: float, crop: np.ndarray, conf: float = 1.0):
        if not self.ratios:
            self.frame_start, self.bbox_first = t, box
        self.frame_end, self.bbox_last = t, box
        self.peak_confidence = max(self.peak_confidence, conf)
        self.ratios.append(ratio)
        # ukuran buah
        w_px = box[2] - box[0]
        h_px = box[3] - box[1]
        self.widths_px.append(w_px)
        self.heights_px.append(h_px)
        self.aspects.append(w_px / max(h_px, 1))
        if score > self.best_score and crop.size:
            self.best_score, self.thumbnail = score, _crop_thumbnail(crop)

    @property
    def measurement(self) -> dict:
        """Ringkasan pengukuran dimensi buah dari berbagai frame/sudut."""
        if not self.widths_px:
            return {}
        return {
            "avg_width_px": float(np.mean(self.widths_px)),
            "avg_height_px": float(np.mean(self.heights_px)),
            "max_width_px": int(np.max(self.widths_px)),
            "max_height_px": int(np.max(self.heights_px)),
            "avg_aspect": float(np.mean(self.aspects)),
            "std_aspect": float(np.std(self.aspects)),
            "n_views": len(self.widths_px),
        }


class MangoPipeline:
    def __init__(self, weights: str = "weights/mango.pt", conf: float = 0.35,
                 imgsz: int = 960, device: str | None = None,
                 ripeness_cfg: rp.RipenessConfig | None = None,
                 classes: list[str] | tuple[str, ...] | None = None):
        self.generic = "yoloe" in Path(weights).name.lower()
        if not Path(weights).exists() and not self.generic:
            raise FileNotFoundError(
                f"Model '{weights}' tidak ditemukan. Latih dulu dengan train.py "
                "atau taruh file .pt hasil training di path tersebut."
            )
        self.model = (YOLOE if self.generic else YOLO)(weights)
        self.conf, self.imgsz, self.device = conf, imgsz, device
        self.cfg = ripeness_cfg or rp.RipenessConfig()
        self.stats: dict[int, TrackRecord] = defaultdict(TrackRecord)
        self.min_hits = 3
        self.min_peak_conf = 0.0
        self.fps_eff = 15.0
        self._t = 0
        self.class_ids: list[int] | None = None
        self.set_classes(classes if classes is not None else (GENERIC_FRUIT if self.generic else None))

    @property
    def vocabulary(self) -> list[str]:
        return sorted(set(self.model.names.values()))

    def set_classes(self, names) -> None:
        if names is None:
            self.class_ids = None
            return
        wanted = {n.lower() for n in names}
        self.class_ids = [i for i, n in self.model.names.items() if n.lower() in wanted]

    # ---------- inti ----------
    def _extract(self, result, frame: np.ndarray) -> list[Fruit]:
        fruits: list[Fruit] = []
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return fruits
        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        ids = boxes.id.cpu().numpy().astype(int) if boxes.id is not None else [None] * len(xyxy)
        polys = result.masks.xy if result.masks is not None else [None] * len(xyxy)
        H, W = frame.shape[:2]
        for (x1, y1, x2, y2), c, tid, k, poly in zip(xyxy, confs, ids, clss, polys):
            x1, y1, x2, y2 = max(x1, 0), max(y1, 0), min(x2, W), min(y2, H)
            crop = frame[y1:y2, x1:x2]
            mask = None
            if poly is not None and len(poly) >= 3 and crop.size:
                mask = np.zeros(crop.shape[:2], np.uint8)
                cv2.fillPoly(mask, [np.round(poly - (x1, y1)).astype(np.int32)], 1)
            r = rp.analyze(crop, self.cfg, mask)
            fruits.append(Fruit((x1, y1, x2, y2), float(c), r, None if tid is None else int(tid),
                                self.model.names.get(int(k), "mango")))
        return fruits

    def _params(self) -> dict:
        kw = dict(conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)
        if self.class_ids is not None:
            kw["classes"] = self.class_ids
        return kw

    def detect_image(self, frame: np.ndarray) -> list[Fruit]:
        """Deteksi satu gambar (tanpa tracking)."""
        if self.class_ids == []:
            return []
        return self._extract(self.model.predict(frame, **self._params())[0], frame)

    def track_frame(self, frame: np.ndarray) -> list[Fruit]:
        """Deteksi + pelacakan antar frame menggunakan ByteTrack.
        ByteTrack dipilih karena efisien: menggunakan dua tahap asosiasi
        (deteksi confidence tinggi lalu rendah) tanpa fitur visual tambahan."""
        self._t += 1
        if self.class_ids == []:
            return []
        res = self.model.track(frame, persist=True, tracker="bytetrack.yaml", **self._params())[0]
        fruits = self._extract(res, frame)
        H, W = frame.shape[:2]
        for f in fruits:
            if f.track_id is None:
                continue
            x1, y1, x2, y2 = (int(v) for v in f.box)
            area = (x2 - x1) * (y2 - y1)
            score = f.conf * area
            # kurangi skor jika buah terpotong tepi frame
            if x1 <= 3 or y1 <= 3 or x2 >= W - 3 or y2 >= H - 3:
                score *= 0.4
            self.stats[f.track_id].record(self._t, f.box, f.ripeness.ratio, score, frame[y1:y2, x1:x2], f.conf)
        return fruits

    def reset_tracking(self):
        self.stats.clear()
        self._t = 0
        if getattr(self.model, "predictor", None) is not None:
            self.model.predictor.trackers = []
            self.model.predictor = None

    # ---------- ringkasan ----------
    @staticmethod
    def summarize(fruits: list[Fruit]) -> Counter:
        return Counter(f.ripeness.label for f in fruits)

    def _iou(self, box_a, box_b) -> float:
        """Hitung Intersection over Union antara dua bounding box."""
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[2], box_b[2])
        y2 = min(box_a[3], box_b[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
        area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
        union = area_a + area_b - inter
        return inter / max(union, 1e-6)

    def merge_tracks(self, min_hits: int | None = None, gap_s: float = 1.2, iou_thresh: float = 0.15) -> list[list[int]]:
        """Gabungkan ID track yang sebenarnya satu buah menggunakan IoU dan kedekatan temporal.

        Pendekatan berbasis IoU: dua track dianggap satu buah jika bbox akhir track pertama
        dan bbox awal track kedua punya IoU >= iou_thresh, muncul dalam jeda waktu < gap_s detik,
        dan ukurannya tidak berubah drastis (rasio luas 0.5-2.0).
        Track dengan frame < min_hits atau confidence puncak < min_peak_conf dibuang."""
        min_hits = self.min_hits if min_hits is None else min_hits
        valid = sorted(((i, s) for i, s in self.stats.items()
                        if len(s.ratios) >= min_hits and s.peak_confidence >= self.min_peak_conf),
                       key=lambda p: p[1].frame_start)
        max_gap_frames = gap_s * self.fps_eff

        # Union-Find untuk penggabungan
        parent: dict[int, int] = {i: i for i, _ in valid}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a, b):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        # Cari pasangan yang bisa digabung
        for j in range(len(valid)):
            idb, B = valid[j]
            for i in range(j):
                ida, A = valid[i]
                if find(ida) == find(idb):
                    continue
                # Cek temporal: A berakhir sebelum B mulai, dalam jeda waktu
                if A.frame_end >= B.frame_start or B.frame_start - A.frame_end > max_gap_frames:
                    continue
                # Cek IoU antara bbox terakhir A dan bbox pertama B
                iou = self._iou(A.bbox_last, B.bbox_first)
                if iou < iou_thresh:
                    continue
                # Cek rasio ukuran
                area_a = (A.bbox_last[2] - A.bbox_last[0]) * (A.bbox_last[3] - A.bbox_last[1])
                area_b = (B.bbox_first[2] - B.bbox_first[0]) * (B.bbox_first[3] - B.bbox_first[1])
                ratio = area_b / max(area_a, 1)
                if 0.5 <= ratio <= 2.0:
                    union(ida, idb)

        # Bentuk grup dari Union-Find
        from collections import defaultdict as dd
        groups_map = dd(list)
        for i, _ in valid:
            groups_map[find(i)].append(i)
        return [sorted(ids, key=lambda x: self.stats[x].frame_start) for ids in groups_map.values()]

    def group_info(self, min_hits: int | None = None) -> list[dict]:
        """Satu entri per mangga unik: ID gabungan, jumlah frame, status, foto, dan pengukuran."""
        out = []
        for n, ids in enumerate(self.merge_tracks(min_hits), 1):
            ratios = [r for i in ids for r in self.stats[i].ratios]
            ratio = float(np.mean(ratios))
            best = max((self.stats[i] for i in ids), key=lambda s: s.best_score)
            # Gabungkan pengukuran dari semua ID
            all_widths = [w for i in ids for w in self.stats[i].widths_px]
            all_heights = [h for i in ids for h in self.stats[i].heights_px]
            all_aspects = [a for i in ids for a in self.stats[i].aspects]
            measurement = {
                "avg_width_px": float(np.mean(all_widths)) if all_widths else 0,
                "avg_height_px": float(np.mean(all_heights)) if all_heights else 0,
                "max_width_px": int(np.max(all_widths)) if all_widths else 0,
                "max_height_px": int(np.max(all_heights)) if all_heights else 0,
                "avg_aspect": float(np.mean(all_aspects)) if all_aspects else 0,
                "n_views": len(all_widths),
            }
            # Klasifikasi ukuran berdasarkan area piksel rata-rata
            avg_area = measurement["avg_width_px"] * measurement["avg_height_px"]
            if avg_area >= 15000:
                size_class = "Besar"
            elif avg_area >= 6000:
                size_class = "Sedang"
            else:
                size_class = "Kecil"
            measurement["size_class"] = size_class
            out.append(dict(
                no=n, ids=ids, frames=len(ratios), ratio=ratio,
                label=rp.label_from_ratio(ratio, self.cfg),
                thumb=best.thumbnail,
                t_start=self.stats[ids[0]].frame_start / self.fps_eff,
                t_end=self.stats[ids[-1]].frame_end / self.fps_eff,
                measurement=measurement
            ))
        return out

    def summarize_tracks(self, min_hits: int | None = None) -> tuple[int, Counter]:
        c: Counter = Counter()
        for g in self.group_info(min_hits):
            c[g["label"]] += 1
        return sum(c.values()), c

    # ---------- pengukuran gambar tunggal ----------
    @staticmethod
    def measure_fruits(fruits: list[Fruit]) -> list[dict]:
        """Hitung dimensi tiap buah dari satu gambar (foto)."""
        measurements = []
        for i, f in enumerate(fruits, 1):
            x1, y1, x2, y2 = f.box
            w_px = x2 - x1
            h_px = y2 - y1
            area_px = w_px * h_px
            aspect = w_px / max(h_px, 1)
            if area_px >= 15000:
                size_class = "Besar"
            elif area_px >= 6000:
                size_class = "Sedang"
            else:
                size_class = "Kecil"
            measurements.append({
                "no": i, "width_px": w_px, "height_px": h_px,
                "area_px": area_px, "aspect_ratio": round(aspect, 2),
                "size_class": size_class
            })
        return measurements

    # ---------- gambar ----------
    def label_of(self, f: Fruit) -> str:
        if f.track_id is not None and f.track_id in self.stats:
            return rp.label_from_ratio(self.stats[f.track_id].avg_ratio, self.cfg)
        return f.ripeness.label

    def draw(self, frame: np.ndarray, fruits: list[Fruit], show_total: int | None = None) -> np.ndarray:
        """Anotasi visual: kotak deteksi + label + panel ringkasan. Desain berbeda dari versi sebelumnya."""
        out = frame.copy()
        H, W = out.shape[:2]
        s = max(0.6, min(W, H) / 720)
        line_w = max(2, round(2 * s))
        font, fs = cv2.FONT_HERSHEY_DUPLEX, 0.45 * s
        text_w = max(1, round(s))

        # Overlay transparan
        overlay = out.copy()
        for f in fruits:
            x1, y1, x2, y2 = f.box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), rp.COLORS.get(self.label_of(f), rp.COLORS[rp.UNKNOWN]), -1)
        out = cv2.addWeighted(overlay, 0.18, out, 0.82, 0)

        # Panel ringkasan di pojok kanan-atas (berbeda dari kiri-atas)
        counts = self.summarize(fruits)
        total_lbl = "TERLIHAT" if show_total is None else "UNIK"
        total_val = len(fruits) if show_total is None else show_total
        rows = [(rp.SIAP_PANEN, counts.get(rp.SIAP_PANEN, 0)),
                (rp.MATANG, counts.get(rp.MATANG, 0)), (rp.SETENGAH, counts.get(rp.SETENGAH, 0)),
                (rp.MENTAH, counts.get(rp.MENTAH, 0))]
        m, pad = round(12 * s), round(10 * s)
        big_fs, row_fs = 1.1 * s, 0.5 * s
        (bw_, bh_), _ = cv2.getTextSize(str(total_val), font, big_fs, max(2, round(2 * s)))
        row_h = round(24 * s)
        pw = max(round(200 * s), bw_ + round(120 * s))
        ph = pad * 2 + bh_ + round(8 * s) + row_h * len(rows)
        px = W - m - pw  # pojok kanan
        panel = out.copy()
        _pill_rect(panel, (px, m), (px + pw, m + ph), (20, 60, 20), round(12 * s))
        out = cv2.addWeighted(panel, 0.75, out, 0.25, 0)
        cv2.putText(out, str(total_val), (px + pad, m + pad + bh_), font, big_fs, (255, 255, 255),
                    max(2, round(2 * s)), cv2.LINE_AA)
        cv2.putText(out, total_lbl, (px + pad + bw_ + round(8 * s), m + pad + bh_), font, 0.45 * s,
                    (180, 220, 180), text_w, cv2.LINE_AA)
        for i, (name, n) in enumerate(rows):
            cy = m + pad + bh_ + round(8 * s) + row_h * i + row_h // 2
            cv2.rectangle(out, (px + pad, cy - round(4 * s)),
                          (px + pad + round(10 * s), cy + round(4 * s)),
                          rp.COLORS[name], -1, cv2.LINE_AA)
            cv2.putText(out, f"{name}", (px + pad + round(18 * s), cy + round(4 * s)), font, row_fs,
                        (220, 240, 220), text_w, cv2.LINE_AA)
            nt = str(n)
            (nw, _), _ = cv2.getTextSize(nt, font, row_fs, max(1, round(1.5 * s)))
            cv2.putText(out, nt, (px + pw - pad - nw, cy + round(4 * s)), font, row_fs, (255, 255, 255),
                        max(1, round(1.5 * s)), cv2.LINE_AA)

        # Kotak tiap buah: garis penuh + label di bawah (bukan sudut + label di atas)
        for f in fruits:
            x1, y1, x2, y2 = f.box
            label = self.label_of(f)
            color = rp.COLORS.get(label, rp.COLORS[rp.UNKNOWN])
            cv2.rectangle(out, (x1, y1), (x2, y2), color, line_w, cv2.LINE_AA)
            tag = f"{label} {f.ripeness.ratio:.0%}"
            (tw, th), base = cv2.getTextSize(tag, font, fs, text_w)
            padx, pady = round(5 * s), round(3 * s)
            bw, bh = tw + 2 * padx, th + base + 2 * pady
            bx = min(max(x1, 0), max(W - bw, 0))
            by = y2 + 2 if y2 + 2 + bh < H else y1 - bh - 2  # di bawah kotak (berbeda)
            _pill_rect(out, (bx, by), (bx + bw, by + bh), color, round(4 * s))
            cv2.putText(out, tag, (bx + padx, by + pady + th), font, fs, (255, 255, 255), text_w, cv2.LINE_AA)
        return out


def _pill_rect(img: np.ndarray, p1, p2, color, r: int):
    """Persegi panjang terisi dengan sudut membulat (pill shape)."""
    x1, y1 = p1
    x2, y2 = p2
    r = max(0, min(r, (x2 - x1) // 2, (y2 - y1) // 2))
    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r), color, -1, cv2.LINE_AA)
    for cx, cy in ((x1 + r, y1 + r), (x2 - r, y1 + r), (x1 + r, y2 - r), (x2 - r, y2 - r)):
        cv2.circle(img, (cx, cy), r, color, -1, cv2.LINE_AA)
