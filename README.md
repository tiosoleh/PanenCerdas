<p align="center"><img src="assets/logo.svg" width="96" alt="Logo PanenCerdas"></p>

# PanenCerdas — Deteksi, Hitung & Ukur Mangga

Sistem cerdas untuk mendeteksi mangga, menghitung jumlah buah per pohon, menganalisis kematangan,
dan mengukur dimensi buah dari berbagai sudut pandang — menggunakan foto, video, atau kamera langsung.

## Teknologi

| Komponen | Teknologi | Alasan |
|---|---|---|
| Deteksi objek | **YOLO26m** (Ultralytics; bobot hasil training PanenCerdas) | Real-time, mudah di-fine-tune |
| Pelacakan | **ByteTrack** + penggabungan IoU | Efisien, dua tahap asosiasi confidence |
| Kematangan | **CIE-LAB** color space | Sumbu a\* (hijau↔merah) lebih perseptual dari HSV |
| Pengukuran | **Multi-frame bbox** analysis | Dimensi buah dari berbagai sudut pandang |
| Antarmuka | **Streamlit** | Upload, kamera, video dalam satu web app |

## Cara Pakai

```bash
pip install -r requirements.txt

# 0) Siapkan dataset (MangoYOLO, CC BY 4.0)
python prepare_dataset.py

# 1) Latih model -> menghasilkan weights/mango.pt (YOLO26m hasil training PanenCerdas)
python train.py --data datasets/mango_yolo/data.yaml --epochs 100

# 2) Jalankan aplikasi web
streamlit run app.py

# atau deteksi langsung via OpenCV
python live.py --source 0
```

## Fitur Utama

### 1. Deteksi & Penghitungan
- **Foto**: deteksi satu gambar, hitung buah per frame
- **Video**: tracking dengan ByteTrack, ID unik per buah, penggabungan ID berbasis IoU
- **Live**: deteksi real-time dari webcam

### 2. Analisis Kematangan (CIE-LAB)
Berbeda dari pendekatan HSV konvensional, sistem ini menggunakan ruang warna **CIE-LAB**:
- Konversi BGR → LAB
- Sumbu **a\*** (hijau↔merah): nilai tinggi = matang, nilai rendah = mentah
- Filter luminansi (L\*) untuk mengabaikan bayangan dan pantulan
- Filter kromatisitas (b\*) untuk memastikan piksel valid
- Status: **Matang** (≥60%), **Setengah matang** (30-60%), **Mentah** (<30%)

### 3. Pengukuran Multi-Arah (Fitur Baru)
- Setiap frame merekam dimensi bbox (lebar × tinggi piksel)
- Video: buah diamati dari banyak frame/sudut → estimasi dimensi lebih akurat
- Rasio aspek menunjukkan bentuk buah (lonjong vs bulat)
- Klasifikasi ukuran: **Kecil** / **Sedang** / **Besar**

### 4. Pelacakan ByteTrack + Penggabungan IoU
- ByteTrack: dua tahap asosiasi (confidence tinggi lalu rendah)
- Penggabungan: dua track dianggap satu buah jika IoU bbox ≥ 0.15,
  jeda waktu < 1.2 detik, dan rasio luas 0.5-2.0
- Union-Find untuk pengelompokan transitif

## Struktur

```
app.py              Web app Streamlit (5 tab: Foto, Kamera, Video, Live, Pengukuran)
assets/logo.svg     Logo PanenCerdas
.streamlit/         Tema warna (config.toml)
live.py             Deteksi langsung via OpenCV
train.py            Fine-tuning YOLO
prepare_dataset.py  Konversi dataset MangoYOLO
mango/pipeline.py   Deteksi + tracking ByteTrack + pengukuran
mango/ripeness.py   Analisis kematangan CIE-LAB
mango/video.py      Proses file video
weights/            Model terlatih
```

## Batasan
- Ambang warna adalah heuristik; beberapa varietas (harum manis) dipanen saat hijau
- Pencahayaan ekstrem menurunkan akurasi
- Pengukuran dalam piksel, bukan cm (perlu kalibrasi untuk satuan metrik)
