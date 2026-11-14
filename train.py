"""Fine-tuning YOLOv8 untuk deteksi mangga.

Dataset format YOLO (images/ + labels/ + data.yaml) dari Roboflow Universe / Kaggle
(cari "mango detection"). Lalu:

  python train.py --data path/ke/data.yaml --epochs 100
"""
import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--base", default="yolov8s.pt", help="yolov8n/s/m.pt (pretrained COCO)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    model = YOLO(a.base)  # transfer learning dari bobot COCO
    model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch,
                hsv_h=0.01, hsv_s=0.5, hsv_v=0.3,   # augmentasi warna dibuat kecil: warna = sinyal kematangan
                mosaic=1.0, fliplr=0.5, project="runs", name="mango")
    metrics = model.val()
    print("mAP50:", metrics.box.map50, "mAP50-95:", metrics.box.map)

    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    Path("weights").mkdir(exist_ok=True)
    shutil.copy(best, "weights/mango.pt")
    print("Disalin ke weights/mango.pt")


if __name__ == "__main__":
    main()
