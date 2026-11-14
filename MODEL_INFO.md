# Model PanenCerdas — YOLO26m

File bobot aplikasi: `weights/mango.pt` (berasal dari `best.pt` hasil training PanenCerdas).

## Ringkasan pelatihan dan evaluasi
- Arsitektur: YOLO26m, satu kelas (`mango`).
- Training: 100 epoch, image size 640, batch 4.
- Dataset: 3.456 gambar (train 2.658, validation 374, test 424).
- Evaluasi test: 424 gambar, 2.321 instance, 75 gambar background.
- Precision: 0.9429 (94.29%).
- Recall: 0.9323 (93.23%).
- mAP@50: 0.9812 (98.12%).
- mAP@50-95: 0.7253 (72.53%).

Metrik di atas merupakan hasil evaluasi yang dilaporkan dari run training/evaluasi PanenCerdas.
Model mendeteksi satu kelas objek `mango`; analisis kematangan pada aplikasi merupakan pipeline terpisah berbasis warna CIE-LAB.
