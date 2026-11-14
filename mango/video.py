"""Proses file video: tracking + anotasi + pengukuran, simpan mp4 (H.264)."""
from __future__ import annotations

from typing import Callable

import cv2
import imageio.v2 as imageio

from .pipeline import MangoPipeline


def process_video(pipe: MangoPipeline, src: str, dst: str,
                  progress: Callable[[float], None] | None = None,
                  min_hits: int = 3, frame_skip: int = 1, min_peak_conf: float = 0.0):
    pipe.reset_tracking()
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise IOError(f"Tidak bisa membuka video: {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    pipe.fps_eff = fps / frame_skip
    pipe.min_hits, pipe.min_peak_conf = min_hits, min_peak_conf
    writer = imageio.get_writer(dst, fps=fps / frame_skip, codec="libx264", macro_block_size=1)
    idx = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if (idx - 1) % frame_skip:
                continue
            fruits = pipe.track_frame(frame)
            n_unique, _ = pipe.summarize_tracks()
            out = pipe.draw(frame, fruits, show_total=n_unique)
            h, w = out.shape[:2]
            out = out[:h // 2 * 2, :w // 2 * 2]
            writer.append_data(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
            if progress:
                progress(min(idx / total_frames, 1.0))
    finally:
        cap.release()
        writer.close()
    return pipe.summarize_tracks()
