import streamlit as st
import tempfile
import os
import json
import time

from utils.model   import load_detector, predict
from utils.gradcam import load_xception, compute_gradcam, compute_zone_scores
from utils.video   import extract_frames
from utils.charts  import (
    show_verdict_card,
    show_metric_cards,
    show_analysis_charts,
    show_score_distribution,
    show_gradcam_grid,
    show_region_scores,
)

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(page_title="DeepScan", page_icon="🔍", layout="wide")

# ─────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Share Tech Mono', monospace;
    background-color: #050a0f;
    color: #c9e0d5;
}
.stApp {
    background: #050a0f;
}

/* Hide streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="stSidebar"] { display: none; }

/* File uploader */
[data-testid="stFileUploader"] {
    background: #080f14;
    border: 1px dashed #1a3a2a;
    border-radius: 4px;
    padding: 1rem;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #00e5ff, #00ff88);
}

/* Buttons */
div.stButton > button {
    background: #00e5ff;
    color: #050a0f;
    font-family: 'Share Tech Mono', monospace;
    font-size: 1rem;
    letter-spacing: 3px;
    border: none;
    border-radius: 2px;
    padding: 0.8rem 3rem;
    width: 100%;
    cursor: pointer;
    transition: all 0.2s ease;
}
div.stButton > button:hover {
    background: #00ff88;
    box-shadow: 0 0 20px rgba(0,229,255,0.4);
}

/* Section labels */
.section-label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    color: #4a7a6a;
    letter-spacing: 3px;
    margin-bottom: 0.5rem;
}

/* Divider */
.divider {
    border: none;
    border-top: 1px solid #0d2018;
    margin: 2rem 0;
}

/* Info badge */
.badge {
    display: inline-block;
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 1px;
    padding: 0.2rem 0.8rem;
    border-radius: 2px;
    margin-right: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────────────────────

detector       = load_detector()
xception_model = load_xception()

# ─────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────

st.markdown("""
<div style="padding: 2rem 0 1rem 0;">
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.75rem;
                color:#00e5ff; letter-spacing:4px; margin-bottom:0.5rem;">
        • AI FORENSICS · ACTIVE
    </div>
    <div style="font-size:3.5rem; font-weight:900; letter-spacing:4px; line-height:1;
                margin-bottom:1rem;">
        <span style="color:#ffffff;">DEEP</span><span style="color:#00e5ff;">SCAN</span>
    </div>
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.85rem;
                color:#4a7a6a; margin-bottom:1.5rem; line-height:1.8;">
        Pixel-level deepfake detection with explainable AI. Grad-CAM heatmaps.<br>
        Ensemble neural models. Zero data retention.
    </div>
    <div>
        <span class="badge" style="background:rgba(0,229,255,0.1); border:1px solid #00e5ff33; color:#00e5ff;">
            MODEL 1 <span style="color:#4a7a6a;">Face-Swap ViT</span>
        </span>
        <span class="badge" style="background:rgba(0,229,255,0.1); border:1px solid #00e5ff33; color:#00e5ff;">
            GRAD-CAM <span style="color:#4a7a6a;">Xception CNN</span>
        </span>
        <span class="badge" style="background:rgba(0,229,255,0.1); border:1px solid #00e5ff33; color:#00e5ff;">
            6 ZONES <span style="color:#4a7a6a;">Facial Region Scores</span>
        </span>
    </div>
</div>
<hr class="divider">
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# UPLOAD + HOW IT WORKS
# ─────────────────────────────────────────────────────────────

col_upload, col_how = st.columns([1.2, 1])

with col_upload:
    st.markdown('<div class="section-label">// INPUT</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.5rem; font-weight:bold; letter-spacing:2px; margin-bottom:1rem;">UPLOAD MEDIA</div>', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"],
        label_visibility="collapsed"
    )

    if uploaded:
        ftype = uploaded.type.split("/")[-1].upper()
        fsize = uploaded.size / 1024
        st.markdown(f"""
        <div style="display:flex; gap:1rem; margin-top:1rem;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem;
                        border:1px solid #1a3a2a; padding:0.3rem 0.8rem; color:#8ab8a8;">
                FILE <span style="color:#ffffff; margin-left:0.5rem;">{uploaded.name}</span>
            </div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem;
                        border:1px solid #1a3a2a; padding:0.3rem 0.8rem; color:#8ab8a8;">
                SIZE <span style="color:#ffffff; margin-left:0.5rem;">{fsize:.1f} KB</span>
            </div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem;
                        border:1px solid #1a3a2a; padding:0.3rem 0.8rem; color:#8ab8a8;">
                TYPE <span style="color:#ffffff; margin-left:0.5rem;">{ftype}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

with col_how:
    st.markdown('<div class="section-label">// PROCESS</div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:1.5rem; font-weight:bold; letter-spacing:2px; margin-bottom:1rem;">HOW IT WORKS</div>', unsafe_allow_html=True)
    steps = [
        ("01", "Upload image or video"),
        ("02", "AI model scores each frame (ViT)"),
        ("03", "Xception Grad-CAM maps which pixels look fake"),
        ("04", "6 facial zones each get a suspicion % score"),
        ("✓",  "No data stored or transmitted"),
    ]
    for num, text in steps:
        color = "#00ff88" if num == "✓" else "#00e5ff"
        st.markdown(f"""
        <div style="display:flex; gap:1rem; align-items:flex-start; margin-bottom:0.6rem;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem;
                        color:{color}; min-width:24px;">{num}</div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem;
                        color:#8ab8a8;">{text}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# RUN SCAN BUTTON
# ─────────────────────────────────────────────────────────────

st.markdown("<br>", unsafe_allow_html=True)

run_scan = False
if uploaded:
    run_scan = st.button("► RUN SCAN")

# ─────────────────────────────────────────────────────────────
# SCAN RESULTS
# ─────────────────────────────────────────────────────────────

if uploaded and run_scan:

    is_video = uploaded.type.startswith("video")
    from PIL import Image as PILImage

    # ══════════════════════════════════════════════════════════
    # IMAGE MODE
    # ══════════════════════════════════════════════════════════

    if not is_video:
        image_pil = PILImage.open(uploaded).convert("RGB")

        # Status bar
        status = st.empty()
        status.markdown("""
        <div style="background:#0d2018; border:1px solid #1a3a2a; border-radius:4px;
                    padding:0.6rem 1rem; font-family:'Share Tech Mono',monospace;
                    font-size:0.8rem; color:#00ff88; margin:1rem 0;">
            ✓ 1 frame ready · Running analysis...
        </div>
        """, unsafe_allow_html=True)

        progress_bar = st.progress(0)

        # Predict
        label, fake_score = predict(image_pil, detector)
        progress_bar.progress(50)

        # Grad-CAM
        blended, cam_resized = compute_gradcam(image_pil, xception_model)
        zone_scores = compute_zone_scores(cam_resized)
        progress_bar.progress(100)
        time.sleep(0.3)
        progress_bar.empty()
        status.empty()

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Verdict
        show_verdict_card(label, fake_score)

        # Metric cards (single image)
        is_fake_frame = 1 if fake_score >= 0.6 else 0
        is_unc_frame  = 1 if 0.4 <= fake_score < 0.6 else 0
        show_metric_cards(fake_score, 1, is_fake_frame, is_unc_frame)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Grad-CAM
        st.markdown('<div class="section-label">// PIXEL-LEVEL ANALYSIS</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:1.2rem; font-weight:bold; letter-spacing:2px; margin-bottom:1rem;">GRAD-CAM ACTIVATION MAP</div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div class="section-label">ORIGINAL</div>', unsafe_allow_html=True)
            st.image(image_pil, use_container_width=True)
        with col2:
            st.markdown('<div class="section-label">GRAD-CAM OVERLAY</div>', unsafe_allow_html=True)
            st.image(blended, use_container_width=True)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Region scores
        st.markdown('<div class="section-label">// FACIAL FORENSICS</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:1.2rem; font-weight:bold; letter-spacing:2px; margin-bottom:0.3rem;">REGION SUSPICION SCORES</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-label">GRAD-CAM ACTIVATION PER FACIAL ZONE</div>', unsafe_allow_html=True)
        show_region_scores(zone_scores)

    # ══════════════════════════════════════════════════════════
    # VIDEO MODE
    # ══════════════════════════════════════════════════════════

    else:
        # Save to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        # Extract frames
        frames, frame_numbers, total_frames, fps = extract_frames(tmp_path, every_nth=15)

        status = st.empty()
        status.markdown(f"""
        <div style="background:#0d2018; border:1px solid #1a3a2a; border-radius:4px;
                    padding:0.6rem 1rem; font-family:'Share Tech Mono',monospace;
                    font-size:0.8rem; color:#00ff88; margin:1rem 0;">
            ✓ {len(frames)} frame(s) ready · {fps:.2f}s · Running ensemble inference...
        </div>
        """, unsafe_allow_html=True)

        progress_bar = st.progress(0)
        progress_label = st.empty()

        # ── Predict all frames ──
        labels_list = []
        scores_list = []
        total_steps = len(frames) * 2  # predict + gradcam per frame

        for i, frame_pil in enumerate(frames):
            lbl, fscore = predict(frame_pil, detector)
            labels_list.append(lbl)
            scores_list.append(fscore)
            progress_bar.progress(int((i + 1) / total_steps * 100))
            progress_label.markdown(
                f'<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.75rem; color:#4a7a6a;">RUNNING ENSEMBLE INFERENCE · frame {frame_numbers[i]}</div>',
                unsafe_allow_html=True
            )

        # ── Compute Grad-CAM for every frame ──
        blended_list    = []
        zone_scores_list = []

        for i, frame_pil in enumerate(frames):
            blended, cam_resized = compute_gradcam(frame_pil, xception_model)
            zone_scores = compute_zone_scores(cam_resized)
            blended_list.append(blended)
            zone_scores_list.append(zone_scores)
            progress_bar.progress(int((len(frames) + i + 1) / total_steps * 100))
            progress_label.markdown(
                f'<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.75rem; color:#4a7a6a;">COMPUTING GRAD-CAM · frame {frame_numbers[i]}</div>',
                unsafe_allow_html=True
            )

        progress_bar.empty()
        progress_label.empty()
        status.empty()

        # Overall verdict
        avg_fake      = sum(scores_list) / len(scores_list)
        overall_label = "Fake" if avg_fake >= 0.5 else "Real"
        fake_count    = sum(1 for s in scores_list if s >= 0.6)
        uncertain     = sum(1 for s in scores_list if 0.4 <= s < 0.6)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Verdict
        show_verdict_card(overall_label, avg_fake)

        # Metric cards
        show_metric_cards(avg_fake, len(frames), fake_count, uncertain)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Analysis charts
        st.markdown('<div class="section-label">// ANALYSIS CHARTS</div>', unsafe_allow_html=True)
        show_analysis_charts(avg_fake, scores_list, frame_numbers)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Score distribution
        st.markdown('<div class="section-label">SCORE DISTRIBUTION</div>', unsafe_allow_html=True)
        show_score_distribution(scores_list)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Grad-CAM grid — all frames
        st.markdown('<div class="section-label">// PIXEL-LEVEL ANALYSIS</div>', unsafe_allow_html=True)
        st.markdown('<div style="font-size:1.2rem; font-weight:bold; letter-spacing:2px; margin-bottom:0.3rem;">GRAD-CAM ACTIVATION MAPS</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="display:flex; gap:1rem; margin-bottom:1rem;">
            <span style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#4a7a6a;">COLOUR KEY ·</span>
            <span style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#00e5ff;">■ REAL</span>
            <span style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#00ff88;">■ LOW</span>
            <span style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#ffaa00;">■ SUSPECT</span>
            <span style="font-family:'Share Tech Mono',monospace; font-size:0.7rem; color:#ff3355;">■ FAKE</span>
        </div>
        """, unsafe_allow_html=True)

        # Sort by fake score descending, keep top 10
        top10 = sorted(
            zip(scores_list, frames, blended_list, frame_numbers, zone_scores_list),
            key=lambda x: x[0],
            reverse=True
        )[:10]
        t_scores, t_frames, t_blended, t_fnums, t_zones = zip(*top10)

        show_gradcam_grid(t_frames, t_blended, t_fnums, t_scores, t_zones)

        st.markdown('<hr class="divider">', unsafe_allow_html=True)

        # Export
        st.markdown('<div class="section-label">// EXPORT</div>', unsafe_allow_html=True)
        report = {
            "overall_label":  overall_label,
            "avg_fake_score": round(avg_fake, 4),
            "total_frames":   len(frames),
            "fake_frames":    fake_count,
            "uncertain":      uncertain,
            "clean":          len(frames) - fake_count - uncertain,
            "frame_scores":   [{"frame": f, "score": round(s, 4), "label": l}
                               for f, s, l in zip(frame_numbers, scores_list, labels_list)]
        }
        st.download_button(
            label="↓ DOWNLOAD REPORT (JSON)",
            data=json.dumps(report, indent=2),
            file_name="deepscan_report.json",
            mime="application/json"
        )

        os.unlink(tmp_path)
