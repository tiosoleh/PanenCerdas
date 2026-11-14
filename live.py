"""Deteksi langsung dari webcam / RTSP / file video. Tekan 'q' untuk berhenti.

Contoh:
  python live.py                       # webcam 0
  python live.py --source 1            # webcam kedua
  python live.py --source rtsp://...   # kamera IP
"""
import argparse

import cv2

from mango.pipeline import MangoPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--weights", default="weights/mango.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()

    src = int(a.source) if a.source.isdigit() else a.source
    pipe = MangoPipeline(a.weights, conf=a.conf, imgsz=a.imgsz)
    pipe.min_hits, pipe.fps_eff = 3, 10.0
    cap = cv2.VideoCapture(src)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        fruits = pipe.track_frame(frame)
        n, _ = pipe.summarize_tracks()
        cv2.imshow("SiMangga (q = keluar)", pipe.draw(frame, fruits, show_total=n))
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()
    total, counts = pipe.summarize_tracks()
    print(f"Total mangga unik: {total}")
    for k, v in counts.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()