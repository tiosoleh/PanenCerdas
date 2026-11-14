"""Aplikasi web SiMangga — streamlit run app.py"""
import base64
import html
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st

from mango import ripeness as rp
from mango.pipeline import GENERIC_FRUIT, MangoPipeline
from mango.video import process_video
from mango.preprocessing import PreprocessConfig, preprocess, get_preprocessing_info
from mango.evaluation import (
    compute_counting_error, compute_harvest_estimation,
    build_evaluation, compute_detection_metrics, compute_error_metrics,
)

ROOT = Path(__file__).parent
LOGO = ROOT / "assets" / "logo.svg"

st.set_page_config(page_title="SiMangga", page_icon=str(LOGO), layout="wide",
                   initial_sidebar_state="expanded")


# --------------------------------------------------------------------------- utilitas
def H(s: str) -> str:
    return "".join(line.strip() for line in s.strip().splitlines())


def hex_of(bgr) -> str:
    b, g, r = bgr
    return f"#{r:02X}{g:02X}{b:02X}"


COLOR = {k: hex_of(v) for k, v in rp.COLORS.items()}
ORDER = (rp.SIAP_PANEN, rp.MATANG, rp.SETENGAH, rp.MENTAH)
LOGO_URI = "data:image/svg+xml;base64," + base64.b64encode(LOGO.read_bytes()).decode()

# --------------------------------------------------------------------------- CSS
CSS = """
<style>
:root{--ink:#1A1A2E;--muted:#64748B;--line:#E2E8F0;--accent:#0F766E;--accent2:#14B8A6;
  --card:#FFFFFF;--bg:#F0FDFA;--surface:#CCFBF1;--deep:#134E4A}
.block-container{padding-top:.8rem;padding-bottom:3rem;max-width:1200px}
header[data-testid="stHeader"]{background:transparent}
#MainMenu,footer{visibility:hidden}

/* sidebar — navigasi mode */
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#134E4A 0%,#0F766E 100%);
  border-right:none;min-width:220px}
section[data-testid="stSidebar"] .block-container{padding-top:.6rem}
section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown span,
section[data-testid="stSidebar"] .stRadio label span{color:#E0F2F1 !important}
section[data-testid="stSidebar"] .stRadio > div{gap:0}
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]{padding:10px 14px;margin:3px 0;
  border-radius:8px;transition:background .15s}
section[data-testid="stSidebar"] .stRadio [data-baseweb="radio"]:hover{background:rgba(255,255,255,.12)}
section[data-testid="stSidebar"] [aria-checked="true"]{background:rgba(255,255,255,.18) !important;border-radius:8px}

.sb-brand{text-align:center;padding:18px 10px 20px;border-bottom:1px solid rgba(255,255,255,.15);margin-bottom:16px}
.sb-brand img{width:48px;height:48px;margin:0 auto 6px}
.sb-brand .name{font-size:1.2rem;font-weight:800;color:#fff;letter-spacing:-.01em}
.sb-brand .desc{font-size:.7rem;color:#A7F3D0;margin-top:2px}
.sb-nav-label{font-size:.6rem;font-weight:700;letter-spacing:.1em;text-transform:uppercase;
  color:#5EEAD4;padding:0 14px;margin:18px 0 6px}
.sb-footer{margin-top:auto;padding:14px;border-top:1px solid rgba(255,255,255,.12);text-align:center}
.sb-footer .v{font-size:.65rem;color:#5EEAD4;font-weight:600}

/* main area — panel pengaturan */
.cfg-panel{background:#fff;border:1px solid #E2E8F0;border-radius:12px;padding:16px 20px;margin-bottom:18px;
  box-shadow:0 1px 3px rgba(15,118,110,.06)}
.cfg-panel .cfg-title{display:flex;align-items:center;gap:8px;margin-bottom:10px}
.cfg-panel .cfg-title .icon{width:28px;height:28px;border-radius:6px;background:#CCFBF1;
  display:flex;align-items:center;justify-content:center;font-size:.85rem}
.cfg-panel .cfg-title h3{margin:0;font-size:.92rem;font-weight:700;color:var(--ink)}

/* header strip */
.page-head{padding:14px 0 10px;margin-bottom:6px;border-bottom:2px solid #CCFBF1}
.page-head-inner{display:flex;align-items:center;gap:14px;flex-wrap:wrap}
.page-head .mode-badge{display:inline-flex;align-items:center;gap:6px;padding:5px 14px;
  background:#0F766E;color:#fff;border-radius:20px;font-size:.78rem;font-weight:700;letter-spacing:.03em}
.page-head .mode-badge .emoji{font-size:1rem}
.page-head .mode-desc{font-size:.82rem;color:var(--muted);margin-left:4px}
.legend-strip{display:flex;gap:14px;margin-left:auto}
.legend-chip{display:flex;align-items:center;gap:5px;font-size:.75rem;font-weight:600;color:var(--ink)}
.legend-dot{width:8px;height:8px;border-radius:2px;flex:none}

/* upload */
[data-testid="stFileUploaderDropzone"]{border:2px dashed #99F6E4;background:#F0FDFA;border-radius:10px;padding:1.2rem}
[data-testid="stFileUploaderDropzone"]:hover{border-color:#0F766E;background:#CCFBF1}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-child{font-size:0}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span:first-child::after{content:"Seret file ke sini";
  font-size:.95rem;font-weight:700;color:var(--ink)}
[data-testid="stFileUploaderDropzoneInstructions"] > div > span:last-child{display:none}
[data-testid="stFileUploaderDropzone"] button{font-size:0;padding:.45rem 1rem}
[data-testid="stFileUploaderDropzone"] button::after{content:"Pilih file";font-size:.85rem;font-weight:700}
[data-testid="stCameraInput"]{max-width:740px}

/* buttons */
.stButton>button,.stDownloadButton>button{border-radius:8px;font-weight:700;padding:.45rem 1rem;
  border:1px solid #E2E8F0;transition:all .15s}
.stButton>button:hover,.stDownloadButton>button:hover{box-shadow:0 3px 12px rgba(15,118,110,.12)}
.stButton>button[kind="primary"],.stDownloadButton>button[kind="primary"]{border:none;
  background:linear-gradient(135deg,#0F766E,#14B8A6);color:#fff;box-shadow:0 2px 10px rgba(15,118,110,.25)}
[data-testid="stImage"] img,[data-testid="stVideo"] video{border-radius:10px;
  box-shadow:0 4px 16px rgba(26,26,46,.08)}
[data-testid="stExpander"]{border:1px solid #E2E8F0;border-radius:8px;background:#fff}

/* ringkasan hasil */
.rbox{background:#fff;border:1px solid #E2E8F0;border-radius:10px;padding:14px 16px;margin-bottom:8px}
.rbox-title{font-size:.65rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:var(--muted)}
.rbox-big{font-size:2.4rem;font-weight:800;line-height:1;color:var(--ink);margin:2px 0}
.rbox-cap{color:var(--muted);font-size:.78rem}
.rbar{display:flex;height:6px;border-radius:99px;overflow:hidden;background:#F1F5F9;margin:10px 0 5px}
.rbar span{display:block;height:100%}
.rline{display:flex;align-items:center;gap:7px;padding:5px 0;border-top:1px solid #F8FAFC;font-size:.82rem}
.rline .n{flex:1;font-weight:600;color:var(--ink)}
.rline .c{font-weight:800;color:var(--ink)}
.rline .p{width:40px;text-align:right;color:var(--muted)}

/* status panen */
.spanen{border-radius:8px;padding:10px 14px;margin-bottom:8px;font-size:.85rem}
.spanen b{display:block;margin-bottom:1px}
.spanen.s-ok{background:#F0FDF4;border:1px solid #BBF7D0;color:#166534}
.spanen.s-mid{background:#FFFBEB;border:1px solid #FDE68A;color:#854D0E}
.spanen.s-low{background:#F0FDFA;border:1px solid #99F6E4;color:#134E4A}

/* catatan */
.note{font-size:.78rem;color:var(--muted);padding:8px 12px;border-radius:6px;background:#F8FAFC;
  border:1px solid #E2E8F0;margin-top:5px}

/* langkah cara pakai — vertical timeline */
.howto-tl{margin-top:14px;padding-left:24px;border-left:3px solid #99F6E4;display:flex;flex-direction:column;gap:16px}
.howto-tl .tl-item{position:relative}
.howto-tl .tl-dot{position:absolute;left:-33px;top:2px;width:18px;height:18px;border-radius:50%;
  background:#0F766E;color:#fff;font-size:.6rem;font-weight:800;display:flex;align-items:center;
  justify-content:center}
.howto-tl h4{margin:0 0 2px;font-size:.88rem;color:var(--ink)}
.howto-tl p{margin:0;color:var(--muted);font-size:.8rem;line-height:1.35}

/* dimensi */
.dim-box{background:#F0FDFA;border:1px solid #99F6E4;border-radius:10px;padding:12px 14px;margin-top:6px}
.dim-title{font-size:.65rem;font-weight:700;letter-spacing:.07em;text-transform:uppercase;color:#0F766E;margin-bottom:5px}
.dim-row{display:flex;gap:5px}
.dim-cell{flex:1;text-align:center;padding:6px;background:#fff;border-radius:6px;border:1px solid #E2E8F0}
.dim-cell .dv{font-size:1.15rem;font-weight:800;color:#134E4A}
.dim-cell .dl{font-size:.62rem;color:#64748B;margin-top:1px}

@media(max-width:820px){.legend-strip{margin-left:0;margin-top:6px}}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------- sidebar — NAVIGASI
MODES = {"Foto": "📷", "Kamera": "📸", "Video": "🎬", "Live": "📡", "Ukur Multi-Sudut": "📐", "Evaluasi": "📊"}

with st.sidebar:
    st.markdown(H(f"""
    <div class="sb-brand"><img src="{LOGO_URI}" alt="">
      <div class="name">SiMangga</div>
      <div class="desc">Identifikasi &amp; Ukur Mangga</div></div>"""), unsafe_allow_html=True)

    st.markdown('<div class="sb-nav-label">Mode Analisis</div>', unsafe_allow_html=True)
    mode = st.radio("mode", list(MODES.keys()), label_visibility="collapsed",
                    format_func=lambda m: f"{MODES[m]}  {m}")

    st.markdown('<div class="sb-nav-label">Pipeline</div>', unsafe_allow_html=True)
    st.caption("OpenCV Preprocessing → YOLO Detection → ByteTrack Tracking → IoU Merge → CIE-LAB Ripeness → Evaluation")
    st.markdown('<div class="sb-nav-label">Algoritma</div>', unsafe_allow_html=True)
    st.caption("• YOLO26m — Object Detection\n• ByteTrack — Multi-Object Tracking\n• Union-Find IoU — Duplicate Removal\n• CIE-LAB — Analisis Kematangan\n• N=|∪ID(Di)| — Unique Counting")
    st.markdown('<div class="sb-footer"><span class="v">SiMangga v3.0</span></div>',
                unsafe_allow_html=True)


# --------------------------------------------------------------------------- model & konfigurasi
PRESETS = {
    "Standar": (0.80, 0.60, 0.30, "Ambang default untuk berbagai varietas."),
    "Harum manis": (0.65, 0.40, 0.18, "Warna kulit tetap hijau saat sudah matang."),
    "Matang penuh": (0.90, 0.75, 0.45, "Hanya buah yang sudah kuning/oranye."),
    "Kustom": None,
}

DETECTORS = {
    "Umum": dict(label="Detektor universal", weights="weights/yoloe-11s-seg-pf.pt", conf=0.25,
                 desc="Mengenali mangga dari bentuk & konteks. Untuk foto siang/sore."),
    "PanenCerdas YOLO26m": dict(label="Model PanenCerdas (YOLO26m)", weights="weights/mango.pt", conf=0.25,
                  desc="Model khusus deteksi mangga hasil training 100 epoch pada dataset gabungan."),
}


@st.cache_resource(show_spinner="Memuat model...")
def load(weights: str) -> MangoPipeline:
    return MangoPipeline(weights)


# --------------------------------------------------------------------------- header utama
st.markdown(H(f"""
<div class="page-head"><div class="page-head-inner">
  <div class="mode-badge"><span class="emoji">{MODES[mode]}</span>{mode}</div>
  <span class="mode-desc">Sistem identifikasi &amp; pengukuran mangga berbasis Computer Vision</span>
  <div class="legend-strip">
    <div class="legend-chip"><span class="legend-dot" style="background:{COLOR[rp.SIAP_PANEN]}"></span>{rp.SIAP_PANEN}</div>
    <div class="legend-chip"><span class="legend-dot" style="background:{COLOR[rp.MATANG]}"></span>{rp.MATANG}</div>
    <div class="legend-chip"><span class="legend-dot" style="background:{COLOR[rp.SETENGAH]}"></span>{rp.SETENGAH}</div>
    <div class="legend-chip"><span class="legend-dot" style="background:{COLOR[rp.MENTAH]}"></span>{rp.MENTAH}</div>
  </div>
</div></div>"""), unsafe_allow_html=True)


# --------------------------------------------------------------------------- panel konfigurasi di main area
with st.container():
    st.markdown('<div class="cfg-panel"><div class="cfg-title">'
                '<div class="icon">⚙️</div><h3>Pengaturan Analisis</h3></div>', unsafe_allow_html=True)
    pc1, pc2, pc3, pc4 = st.columns([1.2, 1, 1, 1.2], gap="medium")

    with pc1:
        det = st.selectbox("Model", list(DETECTORS),
                           format_func=lambda k: f"🔍 {DETECTORS[k]['label']}")
        spec = DETECTORS[det]
        st.caption(spec["desc"])

    with pc2:
        weights = st.text_input("Path bobot", spec["weights"], key=f"w_{det}",
                                help="Ubah jika file model ada di lokasi lain.")
        conf = st.slider("Confidence", 0.05, 0.9, spec["conf"], 0.05, key=f"conf_{det}")

    with pc3:
        imgsz = st.select_slider("Resolusi", [480, 640, 960, 1280], 960)
        preset = st.selectbox("Profil Kematangan", list(PRESETS))

    with pc4:
        if PRESETS[preset]:
            harvest_t, ripe_t, mid_t, desc = PRESETS[preset]
            st.caption(f"Profil: {desc}")
        else:
            harvest_t = st.slider("Batas siap panen", 0.6, 0.98, 0.80, 0.05, key="h_cust")
            ripe_t = st.slider("Batas matang", 0.3, 0.95, 0.60, 0.05, key="r_cust")
            mid_t = st.slider("Batas setengah", 0.05, 0.6, 0.30, 0.05, key="m_cust")
            if mid_t >= ripe_t:
                st.warning("Batas setengah harus < batas matang.")
                mid_t = max(0.05, ripe_t - 0.05)
            if ripe_t >= harvest_t:
                st.warning("Batas matang harus < batas siap panen.")
                ripe_t = max(0.30, harvest_t - 0.05)

    st.markdown('</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------- panel preprocessing
with st.container():
    st.markdown('<div class="cfg-panel"><div class="cfg-title">'
                '<div class="icon">🔧</div><h3>Preprocessing Citra (Section 4)</h3></div>',
                unsafe_allow_html=True)
    pp_enabled = st.checkbox("Aktifkan preprocessing", value=False,
                             help="Terapkan preprocessing OpenCV sebelum deteksi")
    pp_cfg = PreprocessConfig()
    if pp_enabled:
        ppc1, ppc2, ppc3, ppc4 = st.columns(4, gap="medium")
        with ppc1:
            pp_brightness = st.slider("Brightness", -50, 50, 0, 5,
                                       help="Kecerahan: negatif=gelap, positif=terang")
            pp_contrast = st.slider("Contrast", 0.5, 2.5, 1.0, 0.1,
                                     help="Kontras: <1=rendah, >1=tinggi")
        with ppc2:
            pp_sharpen = st.checkbox("Deblur (Unsharp Mask)", value=False)
            if pp_sharpen:
                pp_sharp_amt = st.slider("Kekuatan penajaman", 0.5, 3.0, 1.5, 0.1)
                pp_sharp_sigma = st.slider("Radius (sigma)", 0.5, 3.0, 1.0, 0.1)
            else:
                pp_sharp_amt, pp_sharp_sigma = 1.5, 1.0
        with ppc3:
            pp_clahe = st.checkbox("CLAHE (Adaptive Histogram)", value=False,
                                    help="Pemerataan histogram adaptif untuk pencahayaan tidak merata")
            if pp_clahe:
                pp_clahe_clip = st.slider("CLAHE clip limit", 1.0, 5.0, 2.0, 0.5)
            else:
                pp_clahe_clip = 2.0
            pp_denoise = st.checkbox("Denoise (NLM)", value=False,
                                      help="Non-Local Means denoising untuk foto ISO tinggi")
            if pp_denoise:
                pp_denoise_h = st.slider("Denoise strength", 3, 20, 10, 1)
            else:
                pp_denoise_h = 10
        with ppc4:
            pp_crop = st.slider("Crop tepi (%)", 0, 20, 0, 1,
                                 help="Potong persen dari setiap sisi frame") / 100.0

        pp_cfg = PreprocessConfig(
            brightness=pp_brightness, contrast=pp_contrast,
            sharpen_enabled=pp_sharpen, sharpen_amount=pp_sharp_amt, sharpen_radius=pp_sharp_sigma,
            clahe_enabled=pp_clahe, clahe_clip=pp_clahe_clip,
            denoise_enabled=pp_denoise, denoise_strength=pp_denoise_h,
            crop_pct=pp_crop,
        )
        active_steps = get_preprocessing_info(pp_cfg)
        if active_steps:
            st.caption("Pipeline aktif: " + " → ".join(active_steps))
        else:
            st.caption("Tidak ada langkah preprocessing yang aktif.")

    st.markdown('</div>', unsafe_allow_html=True)

# load model
try:
    pipe = load(weights)
except Exception as e:
    st.error(f"⚠️ Model gagal dimuat: {e}")
    st.code("python prepare_dataset.py\npython train.py --data datasets/mango_yolo/data.yaml --epochs 100",
            language="bash")
    st.stop()

if pipe.generic:
    vocab = pipe.vocabulary
    st.caption("Kelas yang dianggap mangga:")
    chosen = st.multiselect("Kelas", vocab, default=[c for c in GENERIC_FRUIT if c in vocab],
                            key="gcls", label_visibility="collapsed")
    pipe.set_classes(chosen)

pipe.conf, pipe.imgsz = conf, imgsz
pipe.cfg = rp.RipenessConfig(harvest_threshold=harvest_t, ripe_threshold=ripe_t, mid_threshold=mid_t)


# --------------------------------------------------------------------------- komponen UI
def stat_card(total: int, counts, caption: str = "mangga terdeteksi") -> str:
    segs, rows = "", ""
    known = ORDER + ((rp.UNKNOWN,) if counts.get(rp.UNKNOWN, 0) else ())
    for k in known:
        n = counts.get(k, 0)
        share = n / total if total else 0
        if n:
            segs += f'<span style="width:{share * 100:.2f}%;background:{COLOR[k]}"></span>'
        rows += (f'<div class="rline"><span class="legend-dot" style="background:{COLOR[k]}"></span>'
                 f'<span class="n">{k}</span><span class="c">{n}</span>'
                 f'<span class="p">{share:.0%}</span></div>')
    return H(f"""<div class="rbox"><div class="rbox-title">Jumlah</div><div class="rbox-big">{total}</div>
      <div class="rbox-cap">{caption}</div><div class="rbar">{segs}</div>{rows}</div>""")


def verdict_card(counts, total: int) -> str:
    if not total:
        return H('<div class="spanen s-low"><b>Belum ada mangga</b>'
                 'Turunkan confidence atau naikkan resolusi.</div>')
    n_harvest = counts.get(rp.SIAP_PANEN, 0)
    n_matang = counts.get(rp.MATANG, 0)
    n_ready = n_harvest + n_matang  # siap panen + matang
    pct_ready = n_ready / total
    pct_harvest = n_harvest / total
    if pct_harvest >= 0.5:
        cls, title, msg = "s-ok", "Siap panen", f"{n_harvest} buah siap panen ({pct_harvest:.0%}), {n_matang} matang."
    elif pct_ready >= 0.3:
        cls, title, msg = "s-mid", "Panen sebagian", f"{n_harvest} siap panen, {n_matang} matang — petik yang berwarna."
    else:
        cls, title, msg = "s-low", "Belum waktunya", f"Baru {pct_ready:.0%} matang/siap. Tunggu lebih lama."
    # Tambahan formula N_panen
    msg += f" N<sub>panen</sub> = {n_harvest}"
    return H(f'<div class="spanen {cls}"><b>{title}</b>{msg}</div>')


def show_verdict(counts, total, caption="mangga terdeteksi"):
    st.markdown(verdict_card(counts, total), unsafe_allow_html=True)
    st.markdown(stat_card(total, counts, caption), unsafe_allow_html=True)


def measure_panel_html(measurements: list[dict]) -> str:
    if not measurements:
        return ""
    avg_w = np.mean([m["width_px"] for m in measurements])
    avg_h = np.mean([m["height_px"] for m in measurements])
    avg_area = np.mean([m["area_px"] for m in measurements])
    return H(f"""
    <div class="dim-box">
      <div class="dim-title">Dimensi Rata-rata</div>
      <div class="dim-row">
        <div class="dim-cell"><div class="dv">{avg_w:.0f}</div><div class="dl">Lebar px</div></div>
        <div class="dim-cell"><div class="dv">{avg_h:.0f}</div><div class="dl">Tinggi px</div></div>
        <div class="dim-cell"><div class="dv">{avg_area:.0f}</div><div class="dl">Area px²</div></div>
      </div>
    </div>""")


def fruit_table(rows: list[dict], key: str):
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, use_container_width=True, key=key)


def howto_steps(items):
    cards = "".join(f'<div class="tl-item"><div class="tl-dot">{i}</div>'
                    f'<h4>{t}</h4><p>{d}</p></div>'
                    for i, (t, d) in enumerate(items, 1))
    st.markdown(H(f'<div class="howto-tl">{cards}</div>'), unsafe_allow_html=True)


def note_box(text: str):
    st.markdown(f'<div class="note">{text}</div>', unsafe_allow_html=True)


STEPS_FOTO = [("Upload gambar", "Foto pohon mangga format JPG atau PNG."),
              ("Analisis otomatis", "YOLO mendeteksi buah, CIE-LAB menilai warna kulit."),
              ("Hasil lengkap", "Jumlah, kematangan, dan dimensi setiap buah.")]


def run_image(file_bytes: bytes, name: str):
    frame = cv2.imdecode(np.frombuffer(file_bytes, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        st.error("Berkas tidak bisa dibaca sebagai gambar.")
        return

    # Preprocessing (Section 4 UTS)
    if pp_enabled:
        frame_processed = preprocess(frame, pp_cfg)
    else:
        frame_processed = frame

    with st.spinner("Menganalisis..."):
        fruits = pipe.detect_image(frame_processed)
        out = pipe.draw(frame_processed, fruits)
        measurements = pipe.measure_fruits(fruits)
    counts = pipe.summarize(fruits)

    # Tampilkan before/after preprocessing jika aktif
    if pp_enabled and get_preprocessing_info(pp_cfg):
        st.markdown("**Preprocessing diterapkan:**", unsafe_allow_html=True)
        pp_c1, pp_c2 = st.columns(2, gap="small")
        with pp_c1:
            st.caption("Sebelum preprocessing")
            st.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), use_container_width=True)
        with pp_c2:
            st.caption("Setelah preprocessing")
            st.image(cv2.cvtColor(frame_processed, cv2.COLOR_BGR2RGB), use_container_width=True)

    # layout: gambar kiri, ringkasan kanan
    col_img, col_info = st.columns([3, 2], gap="medium")
    with col_img:
        st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), use_container_width=True)
        ok, buf = cv2.imencode(".png", out)
        if ok:
            st.download_button("⬇ Unduh hasil", buf.tobytes(), f"{Path(name).stem}_hasil.png", "image/png")
    with col_info:
        st.markdown(verdict_card(counts, len(fruits)), unsafe_allow_html=True)
        st.markdown(stat_card(len(fruits), counts, "mangga di foto"), unsafe_allow_html=True)
        if measurements:
            st.markdown(measure_panel_html(measurements), unsafe_allow_html=True)

        # Evaluasi singkat untuk foto
        labels = [f.ripeness.label for f in fruits]
        n_harvest = compute_harvest_estimation(labels)
        eval_html = H(f"""<div class="dim-box">
          <div class="dim-title">Estimasi Panen (Section 8)</div>
          <div class="dim-row">
            <div class="dim-cell"><div class="dv">{len(fruits)}</div><div class="dl">Total buah (N)</div></div>
            <div class="dim-cell"><div class="dv">{n_harvest}</div><div class="dl">Siap panen (N<sub>panen</sub>)</div></div>
            <div class="dim-cell"><div class="dv">{n_harvest/max(len(fruits),1):.0%}</div><div class="dl">Rasio panen</div></div>
          </div></div>""")
        st.markdown(eval_html, unsafe_allow_html=True)

    msg = ("Foto hanya menangkap satu sisi — buah tersembunyi tidak terhitung. "
           "Gunakan mode Video untuk cakupan lebih luas.")
    if pipe.generic:
        msg += " Model universal bisa salah mengenali buah mirip — atur filter di pengaturan."
    note_box(msg)

    if fruits:
        with st.expander(f"📋 Rincian {len(fruits)} buah"):
            rows = []
            for i, f in enumerate(sorted(fruits, key=lambda f: f.box[0]), 1):
                m = measurements[i - 1] if i <= len(measurements) else {}
                row = {"#": i, "Kematangan": f.ripeness.label,
                       "Rasio": f"{f.ripeness.ratio:.0%}",
                       "Conf": f"{f.conf:.0%}",
                       "W(px)": m.get("width_px", "-"),
                       "H(px)": m.get("height_px", "-"),
                       "Size": m.get("size_class", "-")}
                rows.append(row)
            fruit_table(rows, f"tbl_{name}")


# --------------------------------------------------------------------------- konten per mode
if mode == "Foto":
    up = st.file_uploader("Upload foto pohon mangga (JPG / PNG)", type=["jpg", "jpeg", "png"],
                          help="Pastikan foto terang dan tajam.")
    if up:
        run_image(up.getvalue(), up.name)
    else:
        howto_steps(STEPS_FOTO)

elif mode == "Kamera":
    snap = st.camera_input("Arahkan ke pohon, lalu ambil foto")
    if snap:
        run_image(snap.getvalue(), "kamera")

elif mode == "Video":
    vid = st.file_uploader("Upload video (MP4, MOV, AVI, MKV)", type=["mp4", "mov", "avi", "mkv"],
                           help="Rekam sambil berjalan pelan mengelilingi pohon.")
    if not vid:
        howto_steps([("Rekam keliling", "Berjalan pelan agar semua sisi pohon terekam."),
                     ("Pelacakan otomatis", "Setiap mangga diberi ID unik antar frame."),
                     ("Hasil lengkap", "Galeri buah + dimensi dari berbagai sudut pandang.")])
    else:
        st.markdown("**Parameter Video**")
        vc1, vc2, vc3 = st.columns(3, gap="medium")
        speed = vc1.select_slider(
            "Sampling", [1, 2, 3, 5], 2,
            format_func=lambda n: {1: "Detail", 2: "Normal", 3: "Cepat", 5: "Sangat cepat"}[n],
            help="Proses setiap N frame.")
        min_hits = vc2.slider("Min. kemunculan", 2, 12, 3,
                              help="Buah di bawah ini diabaikan.")
        peak_conf = vc3.slider("Confidence puncak", 0.25, 0.9, max(0.45, conf), 0.05,
                               help="Buah dihitung jika pernah terdeteksi setinggi ini.")
        sig = (vid.name, vid.size)
        cfg_sig = (det, speed, min_hits, peak_conf, round(ripe_t, 2), round(mid_t, 2), round(conf, 2), imgsz,
                   tuple(pipe.class_ids or ()))

        if st.button("▶ Proses Video", type="primary"):
            with tempfile.TemporaryDirectory() as d:
                src = Path(d) / Path(vid.name).name
                src.write_bytes(vid.getvalue())
                dst = Path(d) / "hasil.mp4"
                bar = st.progress(0.0, "Memproses video...")
                total, counts = process_video(
                    pipe, str(src), str(dst), frame_skip=speed, min_hits=min_hits, min_peak_conf=peak_conf,
                    progress=lambda p: bar.progress(p, f"Memproses... {p:.0%}"))
                bar.empty()
                data = dst.read_bytes()
            groups = pipe.group_info()
            raw_ids = len(pipe.stats)
            # Hitung total deteksi mentah (semua bounding box)
            raw_total_det = sum(len(s.ratios) for s in pipe.stats.values())
            st.session_state["video"] = dict(sig=sig, cfg=cfg_sig, data=data, total=total, counts=counts,
                                             groups=groups, raw_ids=raw_ids,
                                             raw_total_detections=raw_total_det,
                                             raw_confirmed=sum(1 for s in pipe.stats.values()
                                                               if len(s.ratios) >= min_hits))
            pipe.min_peak_conf = 0.0

        res = st.session_state.get("video")
        if res and res["sig"] == sig:
            if res["cfg"] != cfg_sig:
                note_box("Pengaturan berubah. Tekan <b>▶ Proses Video</b> untuk memperbarui.")
            # video kiri, stats kanan
            vid_col, stat_col = st.columns([3, 2], gap="medium")
            with vid_col:
                st.video(res["data"])
                st.download_button("⬇ Unduh video hasil", res["data"], "hasil_mangga.mp4", "video/mp4")
            with stat_col:
                st.markdown(verdict_card(res["counts"], res["total"]), unsafe_allow_html=True)
                st.markdown(stat_card(res["total"], res["counts"], "mangga unik"), unsafe_allow_html=True)
            if res["raw_ids"] > res["total"]:
                note_box(f"Tracker mencatat <b>{res['raw_ids']}</b> ID — setelah merge & filter: "
                         f"<b>{res['total']}</b> mangga unik.")
            groups = res["groups"]
            if groups:
                with st.expander(f"🖼 Galeri {len(groups)} mangga", expanded=True):
                    st.caption("Satu foto per mangga unik.")
                    blank = np.full((118, 118, 3), 230, np.uint8)
                    st.image([cv2.cvtColor(g["thumb"], cv2.COLOR_BGR2RGB) if g["thumb"] is not None else blank
                              for g in groups], width=118,
                             caption=[f"#{g['no']} {g['label']} {g.get('measurement',{}).get('size_class','')}"
                                      for g in groups])
                with st.expander("📊 Tabel detail"):
                    fruit_table([{
                        "No": g["no"], "Status": g["label"],
                        "Kematangan": f"{g['ratio']:.0%}",
                        "Frame": g["frames"],
                        "Waktu": f"{g['t_start']:.1f}-{g['t_end']:.1f} dtk",
                        "Lebar maks": g.get("measurement", {}).get("max_width_px", "-"),
                        "Tinggi maks": g.get("measurement", {}).get("max_height_px", "-"),
                        "Ukuran": g.get("measurement", {}).get("size_class", "-"),
                        "Sudut": g.get("measurement", {}).get("n_views", "-"),
                        "ID merge": ", ".join(map(str, g["ids"]))
                    } for g in groups], "tbl_vid")
        elif vid:
            st.video(vid)

elif mode == "Live":
    st.caption("Deteksi langsung dari webcam — matikan toggle untuk berhenti.")
    if st.toggle("🟢 Aktifkan webcam", key="live_on"):
        pipe.reset_tracking()
        pipe.min_hits, pipe.fps_eff, pipe.min_peak_conf = 3, 10.0, 0.0
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Webcam tidak tersedia.")
        else:
            slot = st.empty()
            info = st.empty()
            while cap.isOpened():
                ok, frame = cap.read()
                if not ok:
                    break
                fruits = pipe.track_frame(frame)
                n, counts = pipe.summarize_tracks()
                slot.image(cv2.cvtColor(pipe.draw(frame, fruits, show_total=n), cv2.COLOR_BGR2RGB),
                           use_container_width=True)
                info.markdown(verdict_card(counts, n) + stat_card(n, counts, "mangga unik"),
                              unsafe_allow_html=True)
            cap.release()
    else:
        howto_steps([("Aktifkan webcam", "Tekan toggle di atas, izinkan akses kamera."),
                     ("Arahkan ke pohon", "Gerakkan kamera pelan ke semua sisi."),
                     ("Pantau real-time", "Jumlah dan status diperbarui otomatis.")])

elif mode == "Ukur Multi-Sudut":
    st.subheader("📐 Pengukuran Multi-Sudut")
    st.markdown("""
    Dimensi buah mangga diukur dari **berbagai frame video**. Semakin banyak sudut pandang,
    semakin akurat estimasi ukuran.

    | Metrik | Penjelasan |
    |--------|-----------|
    | Lebar & Tinggi | Dari bounding box terbesar |
    | Rasio aspek | Lebar / tinggi (>1.3 = lonjong) |
    | Jumlah sudut | Berapa frame buah terlihat |
    | Klasifikasi | Kecil / Sedang / Besar |
    """)

    res_vid = st.session_state.get("video")
    if res_vid and res_vid.get("groups"):
        groups = res_vid["groups"]
        st.success(f"✅ Data {len(groups)} mangga tersedia dari video.")

        measure_data = []
        for g in groups:
            m = g.get("measurement", {})
            measure_data.append({
                "No": g["no"],
                "Status": g["label"],
                "Lebar avg": round(m.get("avg_width_px", 0), 1),
                "Tinggi avg": round(m.get("avg_height_px", 0), 1),
                "Lebar maks": m.get("max_width_px", 0),
                "Tinggi maks": m.get("max_height_px", 0),
                "Aspek": round(m.get("avg_aspect", 0), 2),
                "Sudut": m.get("n_views", 0),
                "Ukuran": m.get("size_class", "-"),
            })
        df = pd.DataFrame(measure_data)
        st.dataframe(df, hide_index=True, use_container_width=True)

        c1, c2, c3 = st.columns(3)
        sizes = [g.get("measurement", {}).get("size_class", "?") for g in groups]
        c1.metric("Besar", sizes.count("Besar"))
        c2.metric("Sedang", sizes.count("Sedang"))
        c3.metric("Kecil", sizes.count("Kecil"))

        avg_aspect = np.mean([g.get("measurement", {}).get("avg_aspect", 1) for g in groups])
        if avg_aspect > 1.3:
            shape = "Rata-rata bentuk **lonjong** (aspek > 1.3)"
        elif avg_aspect > 0.85:
            shape = "Rata-rata bentuk **proporsional** (aspek 0.85-1.3)"
        else:
            shape = "Rata-rata bentuk **tinggi** (aspek < 0.85)"
        st.info(shape)
    else:
        st.info("Proses video dulu di mode **Video** untuk mendapatkan data pengukuran.")
        howto_steps([
            ("Rekam video keliling", "Rekam pohon dari berbagai arah."),
            ("Proses di mode Video", "Data pengukuran otomatis tersedia setelahnya."),
            ("Lihat hasil di sini", "Dimensi, bentuk, dan klasifikasi tiap buah.")
        ])

elif mode == "Evaluasi":
    st.subheader("📊 Evaluasi Akurasi Sistem")
    st.markdown("""
    Evaluasi akurasi berdasarkan **Section 9 UTS** — membandingkan hasil Computer Vision
    dengan penghitungan manual.
    """)

    # ---- Tab evaluasi ----
    eval_tab1, eval_tab2, eval_tab3 = st.tabs(["📝 Hitung Error", "📈 Metrik Deteksi", "📚 Formula & Pipeline"])

    with eval_tab1:
        st.markdown("### Error Penghitungan")
        st.latex(r"Error(\%) = \frac{|N_p - N_a|}{N_a} \times 100\%")
        st.caption("Np = prediksi sistem, Na = jumlah aktual (manual)")

        res_vid = st.session_state.get("video")
        if res_vid:
            n_pred = res_vid["total"]
            n_raw_det = res_vid.get("raw_ids", 0)
            groups = res_vid.get("groups", [])
            labels = [g["label"] for g in groups]
        else:
            n_pred = 0
            n_raw_det = 0
            labels = []

        ev_c1, ev_c2 = st.columns(2, gap="medium")
        with ev_c1:
            st.markdown("**Data dari sistem:**")
            if res_vid:
                st.metric("Total deteksi (semua frame)", n_raw_det)
                st.metric("Setelah duplicate removal (N)", n_pred)
                n_harvest = compute_harvest_estimation(labels)
                st.metric("Siap panen (N_panen)", n_harvest)
                st.latex(rf"N = |{{\cup ID(D_i)}}| = {n_pred}")
                st.latex(rf"N_{{panen}} = \sum_{{j=1}}^{{{n_pred}}} I(M_j = H) = {n_harvest}")
            else:
                st.info("Proses video dulu di mode **Video** untuk data otomatis.")
                n_pred = st.number_input("Atau masukkan jumlah prediksi manual (Np)", 0, 500, 0, key="manual_np")

        with ev_c2:
            st.markdown("**Data penghitungan manual:**")
            n_actual = st.number_input("Jumlah buah aktual (Na)", 0, 500, 0,
                                        help="Hitung manual buah di pohon yang sama.", key="n_actual")

            if n_pred > 0 and n_actual > 0:
                result = compute_counting_error(n_pred, n_actual)
                st.markdown("---")
                st.markdown("### 📊 Hasil Evaluasi")

                rc1, rc2, rc3 = st.columns(3)
                rc1.metric("Error (%)", f"{result['error_pct']:.1f}%")
                rc2.metric("Akurasi (%)", f"{result['accuracy_pct']:.1f}%")
                rc3.metric("Selisih", f"{result['diff']:+d}")

                if result['error_pct'] <= 10:
                    st.success(f"✅ {result['status']} — Error di bawah 10%, akurasi sangat baik!")
                elif result['error_pct'] <= 20:
                    st.warning(f"⚠️ {result['status']} — Error {result['error_pct']:.1f}%, masih bisa diterima.")
                else:
                    st.error(f"❌ {result['status']} — Error tinggi, perlu perbaikan parameter.")

                # Tampilkan perhitungan step by step
                with st.expander("📐 Detail perhitungan"):
                    st.latex(rf"Error = \frac{{|{n_pred} - {n_actual}|}}{{{n_actual}}} \times 100\%")
                    st.latex(rf"Error = \frac{{{abs(n_pred - n_actual)}}}{{{n_actual}}} \times 100\% = {result['error_pct']:.2f}\%")
                    st.latex(rf"Akurasi = 100\% - {result['error_pct']:.2f}\% = {result['accuracy_pct']:.2f}\%")
            elif n_actual > 0 and n_pred == 0:
                st.info("Masukkan Np atau proses video terlebih dahulu.")

    with eval_tab2:
        st.markdown("### Metrik Deteksi Objek")
        st.markdown("""
        Jika Anda memiliki data ground truth (anotasi manual), masukkan nilai TP, FP, FN
        untuk menghitung metrik deteksi standar.
        """)

        mc1, mc2, mc3 = st.columns(3)
        tp = mc1.number_input("True Positive (TP)", 0, 1000, 0,
                               help="Buah yang terdeteksi benar")
        fp = mc2.number_input("False Positive (FP)", 0, 1000, 0,
                               help="Deteksi salah (bukan buah)")
        fn = mc3.number_input("False Negative (FN)", 0, 1000, 0,
                               help="Buah yang tidak terdeteksi")

        if tp > 0 or fp > 0 or fn > 0:
            metrics = compute_detection_metrics(tp, fp, fn)
            st.markdown("---")
            dm1, dm2, dm3 = st.columns(3)
            dm1.metric("Precision", f"{metrics['precision']:.4f}")
            dm2.metric("Recall", f"{metrics['recall']:.4f}")
            dm3.metric("F1-Score", f"{metrics['f1_score']:.4f}")

            st.latex(rf"Precision = \frac{{TP}}{{TP + FP}} = \frac{{{tp}}}{{{tp + fp}}} = {metrics['precision']:.4f}")
            st.latex(rf"Recall = \frac{{TP}}{{TP + FN}} = \frac{{{tp}}}{{{tp + fn}}} = {metrics['recall']:.4f}")
            st.latex(rf"F1 = \frac{{2 \times P \times R}}{{P + R}} = {metrics['f1_score']:.4f}")

        st.markdown("---")
        st.markdown("### MAE & RMSE (Multi-Pohon)")
        st.markdown("Masukkan data beberapa pohon untuk menghitung MAE dan RMSE.")

        n_trees = st.number_input("Jumlah pohon/sampel", 1, 20, 3, key="n_trees")
        tree_preds = []
        tree_actuals = []
        tree_cols = st.columns(min(n_trees, 5))
        for i in range(n_trees):
            col = tree_cols[i % len(tree_cols)]
            with col:
                p = st.number_input(f"Prediksi pohon {i+1}", 0, 500, 0, key=f"tp_{i}")
                a = st.number_input(f"Aktual pohon {i+1}", 0, 500, 0, key=f"ta_{i}")
                tree_preds.append(p)
                tree_actuals.append(a)

        if any(a > 0 for a in tree_actuals):
            err_metrics = compute_error_metrics(tree_preds, tree_actuals)
            st.markdown("---")
            em1, em2, em3 = st.columns(3)
            em1.metric("MAE", f"{err_metrics['mae']:.2f}" if err_metrics['mae'] is not None else "-")
            em2.metric("RMSE", f"{err_metrics['rmse']:.2f}" if err_metrics['rmse'] is not None else "-")
            em3.metric("MAPE", f"{err_metrics['mape']:.1f}%" if err_metrics['mape'] is not None else "-")

            st.latex(r"MAE = \frac{1}{n} \sum_{i=1}^{n} |P_i - A_i|")
            st.latex(r"RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (P_i - A_i)^2}")
            st.latex(r"MAPE = \frac{100}{n} \sum_{i=1}^{n} \frac{|P_i - A_i|}{A_i}")

    with eval_tab3:
        st.markdown("### 🔬 Arsitektur Pipeline Sistem")
        st.markdown("""
        Pipeline SiMangga mengikuti arsitektur **8 tahap** sesuai sistem deteksi mangga:

        | No | Tahap | Metode | Keterangan |
        |----|-------|--------|------------|
        | 1 | Akuisisi Citra | Kamera/Upload | Foto/video pohon mangga |
        | 2 | Preprocessing | OpenCV | Resize, deblur, brightness, CLAHE, denoise |
        | 3 | Deteksi Objek | YOLO v11 | Bounding box + confidence + class |
        | 4 | Tracking | ByteTrack | Multi-object tracking antar frame |
        | 5 | Duplicate Removal | Union-Find + IoU | Merge ID yang sama: N = |∪ID(Di)| |
        | 6 | Klasifikasi Kematangan | CIE-LAB (a*, b*) | 4 kelas: Mentah → Setengah → Matang → Siap Panen |
        | 7 | Pengukuran | Bounding Box Multi-View | Dimensi & klasifikasi ukuran |
        | 8 | Estimasi Panen | Fungsi Indikator | N_panen = ΣI(Mj = H) |
        """)

        st.markdown("### 📐 Formula Utama")
        st.latex(r"N = |\cup ID(D_i)| \quad \text{(Unique fruit counting)}")
        st.latex(r"N_{panen} = \sum_{j=1}^{N} I(M_j = H) \quad \text{(Harvest estimation)}")
        st.latex(r"Error(\%) = \frac{|N_p - N_a|}{N_a} \times 100\% \quad \text{(Counting error)}")
        st.latex(r"IoU = \frac{|A \cap B|}{|A \cup B|} \quad \text{(Intersection over Union)}")

        st.markdown("### 🧪 Algoritma yang Digunakan")
        algo_data = [
            {"Algoritma": "YOLO v11", "Fungsi": "Object Detection", "Library": "Ultralytics"},
            {"Algoritma": "ByteTrack", "Fungsi": "Multi-Object Tracking", "Library": "Ultralytics"},
            {"Algoritma": "Union-Find + IoU", "Fungsi": "Duplicate Counting Removal", "Library": "Custom"},
            {"Algoritma": "CIE-LAB Color Analysis", "Fungsi": "Klasifikasi Kematangan", "Library": "OpenCV"},
            {"Algoritma": "Unsharp Mask", "Fungsi": "Deblur / Penajaman", "Library": "OpenCV"},
            {"Algoritma": "CLAHE", "Fungsi": "Histogram Equalization", "Library": "OpenCV"},
            {"Algoritma": "Non-Local Means", "Fungsi": "Denoising", "Library": "OpenCV"},
            {"Algoritma": "Bounding Box Analysis", "Fungsi": "Pengukuran Dimensi", "Library": "Custom"},
        ]
        st.dataframe(pd.DataFrame(algo_data), hide_index=True, use_container_width=True)

        st.markdown("### 📚 Referensi Ilmiah")
        st.markdown("""
        1. Stein et al. (2016) — *Image Based Mango Fruit Detection, Localisation and Yield Estimation Using Multiple View Geometry*. Sensors, 16(11). DOI: 10.3390/s16111915
        2. MangoYOLO — *Mango Fruit Load Estimation Using Video Based MangoYOLO—Kalman Filter—Hungarian Algorithm Method* (2019). Sensors, 19(12). DOI: 10.3390/s19122742
        3. Schönberger & Frahm (2016) — *Structure-from-Motion Revisited*. CVPR.
        4. Payne et al. (2013) — *Estimation of mango crop yield using image analysis*. Computers and Electronics in Agriculture, 91.
        """)