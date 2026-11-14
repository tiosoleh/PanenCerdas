"""Unduh MangoYOLO (CQUniversity, CC BY 4.0) lalu ubah anotasi PASCAL VOC -> format YOLO.

  python prepare_dataset.py

Hasil: datasets/mango_yolo/data.yaml  (dipakai langsung oleh train.py)
Unduhan bisa dilanjutkan: kalau terputus, jalankan ulang perintah yang sama.

Sumber : https://acquire.cqu.edu.au/articles/dataset/MangoYOLO_data_set/13450661
Sitasi : Koirala, Walsh, Wang, McCarthy (2019) "Deep learning for real-time fruit detection
         and orchard fruit load estimation: benchmarking of MangoYOLO", Precision Agriculture.
"""
import argparse
import hashlib
import random
import shutil
import urllib.request
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

URL = "https://ndownloader.figshare.com/files/26220632"
MD5 = "6702f497522098e0068bf39eb3bc79e9"
SIZE = 162591604
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp"}


def download(url: str, dst: Path, size: int, retries: int = 50):
    dst.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(1, retries + 1):
        have = dst.stat().st_size if dst.exists() else 0
        if have >= size:
            return
        print(f"[unduh] percobaan {attempt}: {have / 1e6:.1f}/{size / 1e6:.1f} MB")
        req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r, open(dst, "ab" if have else "wb") as f:
                if have and r.status != 206:  # server mengabaikan Range -> mulai ulang
                    f.close(); dst.unlink(); continue
                while chunk := r.read(1 << 16):
                    f.write(chunk)
        except Exception as e:  # putus koneksi -> lanjut dari byte terakhir
            print("   terputus:", e)
    raise RuntimeError("Gagal mengunduh setelah banyak percobaan; coba lagi nanti / pakai jaringan lain.")


def md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def read_split_lists(root: Path) -> dict[str, set[str]]:
    """Cari file daftar split (train.txt/val.txt/test.txt) bila ada."""
    out = {}
    for name, key in (("train", "train"), ("trainval", "train"), ("val", "val"), ("valid", "val"), ("test", "test")):
        for p in root.rglob(f"{name}.txt"):
            stems = {Path(l.split()[0]).stem for l in p.read_text().splitlines() if l.strip()}
            if stems:
                out.setdefault(key, set()).update(stems)
    return out if "train" in out else {}


def voc_to_yolo(xml_path: Path):
    r = ET.parse(xml_path).getroot()
    W, H = int(r.findtext("size/width")), int(r.findtext("size/height"))
    lines = []
    for o in r.iter("object"):
        b = o.find("bndbox")
        x1, y1, x2, y2 = (float(b.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax"))
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)
        if x2 <= x1 or y2 <= y1:
            continue
        lines.append(f"0 {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
    return lines


def convert(src: Path, out: Path, seed: int = 0):
    images = {p.stem: p for p in src.rglob("*") if p.suffix.lower() in IMG_EXT}
    pairs = [(x, images[x.stem]) for x in sorted(src.rglob("*.xml")) if x.stem in images]
    if not pairs:
        raise RuntimeError("Tidak ada pasangan gambar+XML ditemukan.")
    splits = read_split_lists(src)
    if splits:
        print("[konversi] memakai pembagian split bawaan dataset")
        assign = {}
        for k, stems in splits.items():
            for s in stems:
                assign.setdefault(s, k)
        default = "train"
    else:
        print("[konversi] split tidak ditemukan -> acak 80/10/10")
        random.Random(seed).shuffle(pairs)
        n = len(pairs)
        assign = {p[0].stem: ("train" if i < .8 * n else "val" if i < .9 * n else "test") for i, p in enumerate(pairs)}
        default = "train"

    if out.exists():
        shutil.rmtree(out)
    count = {"train": 0, "val": 0, "test": 0}
    boxes = 0
    for xml, img in pairs:
        sp = assign.get(xml.stem, default)
        lines = voc_to_yolo(xml)
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
        shutil.copy(img, out / "images" / sp / img.name)
        (out / "labels" / sp / f"{img.stem}.txt").write_text("\n".join(lines))
        count[sp] += 1
        boxes += len(lines)
    has_test = count["test"] > 0
    yaml = f"path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\n"
    yaml += "test: images/test\n" if has_test else ""
    yaml += "names:\n  0: mango\n"
    (out / "data.yaml").write_text(yaml)
    print(f"[konversi] selesai: {count}, total kotak: {boxes}\n[konversi] -> {out / 'data.yaml'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="datasets/raw_mangoyolo")
    ap.add_argument("--out", default="datasets/mango_yolo")
    a = ap.parse_args()
    raw = Path(a.raw)
    zpath = raw / "MangoYOLO.zip"
    download(URL, zpath, SIZE)
    if md5(zpath) != MD5:
        raise RuntimeError("MD5 tidak cocok; hapus MangoYOLO.zip lalu ulangi.")
    ext = raw / "extracted"
    if not ext.exists():
        print("[ekstrak]", zpath)
        with zipfile.ZipFile(zpath) as z:
            z.extractall(ext)
    convert(ext, Path(a.out))
    print("\nLanjut: python train.py --data", Path(a.out) / "data.yaml")


if __name__ == "__main__":
    main()
