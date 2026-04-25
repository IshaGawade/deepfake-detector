import streamlit as st
import tempfile
import os

from utils.model import load_detector, predict
from utils.gradcam import load_xception, compute_gradcam
from utils.video import extract_frames
from utils.charts import (
    show_verdict_card,
    show_gauge_chart,
    show_timeline,
    show_bar_chart,
    show_suspicious_frames,
    show_gradcam_display
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Deepfake Detector",
    page_icon="🔍",
    layout="wide"
)

# ─────────────────────────────────────────────────────────────
# GLOBAL CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Rajdhani:wght@400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Rajdhani', sans-serif;
        background-color: #0a0e1a;
        color: #c9d1d9;
    }

    .stApp {
        background: linear-gradient(135deg, #0a0e1a 0%, #0d1b2a 50%, #0a0e1a 100%);
    }

    h1, h2, h3 {
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 2px;
    }

    .main-title {
        font-family: 'Share Tech Mono', monospace;
        font-size: 2.4rem;
        letter-spacing: 4px;
        text-align: center;
        padding: 1.2rem 0 0.3rem 0;
        background: linear-gradient(90deg, #00f5c4, #00aaff, #ff4b6e);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    .subtitle {
        text-align: center;
        font-size: 1rem;
        color: #4a6fa5;
        letter-spacing: 3px;
        margin-bottom: 2rem;
        font-family: 'Share Tech Mono', monospace;
    }

    .verdict-fake {
        background: linear-gradient(135deg, #1a0a0a, #2d0f0f);
        border: 2px solid #ff4b6e;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(255,75,110,0.3), inset 0 0 30px rgba(255,75,110,0.05);
        margin: 1rem 0;
    }

    .verdict-real {
        background: linear-gradient(135deg, #0a1a0f, #0f2d1a);
        border: 2px solid #00f5c4;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(0,245,196,0.3), inset 0 0 30px rgba(0,245,196,0.05);
        margin: 1rem 0;
    }

    .verdict-label-fake {
        font-family: 'Share Tech Mono', monospace;
        font-size: 4rem;
        color: #ff4b6e;
        letter-spacing: 8px;
        text-shadow: 0 0 20px rgba(255,75,110,0.8);
        line-height: 1;
    }

    .verdict-label-real {
        font-family: 'Share Tech Mono', monospace;
        font-size: 4rem;
        color: #00f5c4;
        letter-spacing: 8px;
        text-shadow: 0 0 20px rgba(0,245,196,0.8);
        line-height: 1;
    }

    .confidence-text {
        font-size: 1.4rem;
        letter-spacing: 2px;
        margin-top: 0.8rem;
        color: #8b9ab5;
        font-family: 'Share Tech Mono', monospace;
    }

    .section-header {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.85rem;
        letter-spacing: 3px;
        color: #4a6fa5;
        border-bottom: 1px solid #1e2d42;
        padding-bottom: 0.5rem;
        margin: 1.5rem 0 1rem 0;
        text-transform: uppercase;
    }

    .frame-caption {
        font-family: 'Share Tech Mono', monospace;
        font-size: 0.72rem;
        letter-spacing: 1px;
        text-align: center;
    }

    .fake-caption { color: #ff4b6e; }
    .real-caption { color: #00f5c4; }

    .stProgress > div > div {
        background: linear-gradient(90deg, #00aaff, #00f5c4);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0d1b2a, #0a1220);
        border-right: 1px solid #1e2d42;
    }

    [data-testid="stSidebar"] .stRadio label {
        color: #8b9ab5;
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 1px;
    }

    div.stButton > button {
        background: linear-gradient(135deg, #00aaff22, #00f5c422);
        border: 1px solid #00aaff55;
        color: #00f5c4;
        font-family: 'Share Tech Mono', monospace;
        letter-spacing: 2px;
        border-radius: 6px;
        transition: all 0.3s ease;
    }

    div.stButton > button:hover {
        background: linear-gradient(135deg, #00aaff44, #00f5c444);
        border-color: #00f5c4;
        box-shadow: 0 0 15px rgba(0,245,196,0.3);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────

detector = load_detector()
xception_model = load_xception()

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="main-title">DEEPFAKE DETECTOR</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">[ Explainable AI · Vision Transformer · Grad-CAM ]</div>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem;
                color:#4a6fa5; letter-spacing:2px; margin-bottom:1rem;">
    ANALYSIS MODE
    </div>
    """, unsafe_allow_html=True)

    mode = st.radio("", ["🖼  Image", "🎬  Video"], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem;
                color:#2a3a55; letter-spacing:1px; line-height:1.8;">
    MODEL<br>
    <span style="color:#4a6fa5;">dima806 ViT</span><br>
    Deepfake detection<br><br>
    HEATMAP<br>
    <span style="color:#4a6fa5;">Xception (timm)</span><br>
    Grad-CAM attention
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════
# IMAGE MODE
# ══════════════════════════════════════════════════════════════

if "Image" in mode:
    st.markdown(
        '<div class="section-header">Upload Image</div>',
        unsafe_allow_html=True
    )
    uploaded = st.file_uploader(
        "", type=["jpg", "jpeg", "png"],
        label_visibility="collapsed"
    )

    if uploaded:
        from PIL import Image
        image_pil = Image.open(uploaded).convert("RGB")

        col_img, col_results = st.columns([1, 1.4])

        with col_img:
            st.markdown(
                '<div class="section-header">Uploaded Image</div>',
                unsafe_allow_html=True
            )
            st.image(image_pil, use_container_width=True)

        with col_results:
            with st.spinner("Analyzing image..."):
                label, fake_score = predict(image_pil, detector)

            st.markdown(
                '<div class="section-header">Verdict</div>',
                unsafe_allow_html=True
            )
            show_verdict_card(label, fake_score)

            st.markdown(
                '<div class="section-header">Probability Gauge</div>',
                unsafe_allow_html=True
            )
            show_gauge_chart(fake_score)

        # Grad-CAM — full width below
        st.markdown(
            '<div class="section-header">Grad-CAM Explainability Heatmap</div>',
            unsafe_allow_html=True
        )
        with st.spinner("Computing Grad-CAM heatmap..."):
            try:
                blended = compute_gradcam(image_pil, xception_model)
                show_gradcam_display(image_pil, blended)
            except Exception as e:
                st.warning(f"Grad-CAM could not be computed: {e}")

# ══════════════════════════════════════════════════════════════
# VIDEO MODE
# ══════════════════════════════════════════════════════════════

elif "Video" in mode:
    st.markdown(
        '<div class="section-header">Upload Video</div>',
        unsafe_allow_html=True
    )
    uploaded = st.file_uploader(
        "", type=["mp4", "avi", "mov"],
        label_visibility="collapsed"
    )

    if uploaded:
        # Save uploaded video to a temp file for OpenCV
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        # ── Extract frames ──
        frames, frame_numbers, total_frames, fps = extract_frames(tmp_path, every_nth=15)

        st.info(
            f"Extracted **{len(frames)} frames** from {total_frames} total "
            f"(every 15th frame) at {fps:.1f} fps"
        )

        # ── Analyze each frame with progress bar ──
        st.markdown(
            '<div class="section-header">Analyzing Frames</div>',
            unsafe_allow_html=True
        )
        progress_bar = st.progress(0)
        status_text = st.empty()

        labels_list = []
        scores_list = []

        for i, frame_pil in enumerate(frames):
            lbl, fscore = predict(frame_pil, detector)
            labels_list.append(lbl)
            scores_list.append(fscore)

            pct = int((i + 1) / len(frames) * 100)
            progress_bar.progress(pct)
            status_text.markdown(
                f'<div style="font-family:\'Share Tech Mono\',monospace; '
                f'font-size:0.8rem; color:#4a6fa5;">'
                f'Analyzing frame {frame_numbers[i]} '
                f'({i+1}/{len(frames)})</div>',
                unsafe_allow_html=True
            )

        progress_bar.empty()
        status_text.empty()

        # ── Overall verdict (average fake score) ──
        avg_fake = sum(scores_list) / len(scores_list)
        overall_label = "Fake" if avg_fake >= 0.5 else "Real"

        # ── Verdict + Gauge side by side ──
        col_v, col_g = st.columns(2)
        with col_v:
            st.markdown(
                '<div class="section-header">Overall Verdict</div>',
                unsafe_allow_html=True
            )
            show_verdict_card(overall_label, avg_fake)
        with col_g:
            st.markdown(
                '<div class="section-header">Average Fake Probability</div>',
                unsafe_allow_html=True
            )
            show_gauge_chart(avg_fake)

        # ── Timeline ──
        st.markdown(
            '<div class="section-header">Timeline Analysis</div>',
            unsafe_allow_html=True
        )
        show_timeline(frame_numbers, scores_list)

        # ── Bar chart ──
        st.markdown(
            '<div class="section-header">Frame Classification Breakdown</div>',
            unsafe_allow_html=True
        )
        show_bar_chart(scores_list)

        # ── Suspicious frames grid ──
        st.markdown(
            '<div class="section-header">Top 5 Most Suspicious Frames</div>',
            unsafe_allow_html=True
        )
        show_suspicious_frames(frames, scores_list, frame_numbers)

        # ── Grad-CAM on worst frame only ──
        worst_idx = scores_list.index(max(scores_list))
        worst_frame = frames[worst_idx]
        worst_label = labels_list[worst_idx]

        st.markdown(
            '<div class="section-header">Grad-CAM — Most Suspicious Frame</div>',
            unsafe_allow_html=True
        )
        st.markdown(
            f'<div style="font-family:\'Share Tech Mono\',monospace; '
            f'font-size:0.8rem; color:#4a6fa5; margin-bottom:0.5rem;">'
            f'Frame {frame_numbers[worst_idx]} — '
            f'{scores_list[worst_idx]*100:.1f}% fake probability</div>',
            unsafe_allow_html=True
        )

        with st.spinner("Computing Grad-CAM heatmap..."):
            try:
                blended = compute_gradcam(worst_frame, xception_model)
                show_gradcam_display(worst_frame, blended)
            except Exception as e:
                st.warning(f"Grad-CAM could not be computed: {e}")

        # Cleanup temp file
        os.unlink(tmp_path)
