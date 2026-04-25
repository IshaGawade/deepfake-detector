import streamlit as st
import plotly.graph_objects as go
from PIL import Image

# ─────────────────────────────────────────────────────────────
# Visualization 1 — Verdict Card
# ─────────────────────────────────────────────────────────────

def show_verdict_card(label, fake_score):
    """
    Displays a bold styled verdict card.
    Red for FAKE, green for REAL, with confidence percentage.

    Args:
        label      : str   — "Fake" or "Real"
        fake_score : float — fake probability (0.0 to 1.0)
    """
    conf_pct = fake_score * 100 if label.lower() == "fake" else (1 - fake_score) * 100

    if label.lower() == "fake":
        st.markdown(f"""
        <div class="verdict-fake">
            <div class="verdict-label-fake">⚠ FAKE</div>
            <div class="confidence-text">{conf_pct:.1f}% confident this is a deepfake</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="verdict-real">
            <div class="verdict-label-real">✓ REAL</div>
            <div class="confidence-text">{conf_pct:.1f}% confident this is authentic</div>
        </div>
        """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Visualization 2 — Probability Gauge Chart
# ─────────────────────────────────────────────────────────────

def show_gauge_chart(fake_score):
    """
    Displays a circular Plotly gauge showing fake probability.
    Red zone above 50%, green zone below 50%.
    Decision boundary marker at 50%.

    Args:
        fake_score : float — fake probability (0.0 to 1.0)
    """
    fake_pct = fake_score * 100
    color = "#ff4b6e" if fake_pct >= 50 else "#00f5c4"

    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=fake_pct,
        number={
            'suffix': '%',
            'font': {'size': 36, 'color': color, 'family': 'Share Tech Mono'}
        },
        delta={
            'reference': 50,
            'increasing': {'color': '#ff4b6e'},
            'decreasing': {'color': '#00f5c4'}
        },
        gauge={
            'axis': {
                'range': [0, 100],
                'tickwidth': 1,
                'tickcolor': "#2a3a55",
                'tickfont': {'color': '#4a6fa5', 'family': 'Share Tech Mono', 'size': 11}
            },
            'bar': {'color': color, 'thickness': 0.25},
            'bgcolor': '#0d1b2a',
            'borderwidth': 2,
            'bordercolor': '#1e2d42',
            'steps': [
                {'range': [0, 50],   'color': 'rgba(0,245,196,0.08)'},
                {'range': [50, 100], 'color': 'rgba(255,75,110,0.08)'}
            ],
            'threshold': {
                'line': {'color': '#00aaff', 'width': 3},
                'thickness': 0.85,
                'value': 50
            }
        },
        title={
            'text': "FAKE PROBABILITY",
            'font': {'size': 13, 'color': '#4a6fa5', 'family': 'Share Tech Mono'}
        }
    ))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': '#c9d1d9'},
        height=300,
        margin=dict(l=30, r=30, t=60, b=20)
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Visualization 3 — Timeline Graph (video only)
# ─────────────────────────────────────────────────────────────

def show_timeline(frame_numbers, fake_scores):
    """
    Line chart of fake probability across video frames.
    Red markers = fake frames, green markers = real frames.
    Dashed line at 50% = decision boundary.

    Args:
        frame_numbers : list of int   — original frame indices
        fake_scores   : list of float — fake probability per frame
    """
    fake_pcts = [s * 100 for s in fake_scores]
    colors = ['#ff4b6e' if p >= 50 else '#00f5c4' for p in fake_pcts]

    fig = go.Figure()

    # Shaded danger zone above 50%
    fig.add_hrect(
        y0=50, y1=100,
        fillcolor="rgba(255,75,110,0.05)",
        line_width=0
    )

    # Main line + markers
    fig.add_trace(go.Scatter(
        x=frame_numbers,
        y=fake_pcts,
        mode='lines+markers',
        line=dict(color='#00aaff', width=2, shape='spline'),
        marker=dict(color=colors, size=8, line=dict(color='#0a0e1a', width=1)),
        name='Fake Probability',
        hovertemplate='Frame %{x}<br>Fake prob: %{y:.1f}%<extra></extra>'
    ))

    # Decision boundary
    fig.add_hline(
        y=50,
        line_dash="dash",
        line_color="#00aaff",
        line_width=1.5,
        annotation_text="Decision boundary (50%)",
        annotation_font=dict(color='#00aaff', family='Share Tech Mono', size=11),
        annotation_position="top right"
    )

    fig.update_layout(
        title=dict(
            text='FRAME-BY-FRAME ANALYSIS',
            font=dict(family='Share Tech Mono', size=13, color='#4a6fa5')
        ),
        xaxis=dict(
            title='Frame Number',
            color='#4a6fa5',
            gridcolor='#1e2d42',
            tickfont=dict(family='Share Tech Mono')
        ),
        yaxis=dict(
            title='Fake Probability (%)',
            range=[0, 100],
            color='#4a6fa5',
            gridcolor='#1e2d42',
            tickfont=dict(family='Share Tech Mono')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,27,42,0.6)',
        font=dict(color='#c9d1d9'),
        height=350,
        margin=dict(l=20, r=20, t=50, b=20),
        showlegend=False
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Visualization 4 — Real vs Fake Bar Chart (video only)
# ─────────────────────────────────────────────────────────────

def show_bar_chart(fake_scores):
    """
    Two-bar chart showing count of real vs fake frames.
    Green = real, red = fake.

    Args:
        fake_scores : list of float — fake probability per frame
    """
    fake_count = sum(1 for s in fake_scores if s >= 0.5)
    real_count = len(fake_scores) - fake_count

    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=['Real Frames', 'Fake Frames'],
        y=[real_count, fake_count],
        marker=dict(
            color=['rgba(0,245,196,0.7)', 'rgba(255,75,110,0.7)'],
            line=dict(color=['#00f5c4', '#ff4b6e'], width=2)
        ),
        text=[real_count, fake_count],
        textposition='outside',
        textfont=dict(family='Share Tech Mono', color='#c9d1d9', size=16),
        hovertemplate='%{x}: %{y} frames<extra></extra>'
    ))

    fig.update_layout(
        title=dict(
            text='FRAME CLASSIFICATION BREAKDOWN',
            font=dict(family='Share Tech Mono', size=13, color='#4a6fa5')
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(13,27,42,0.6)',
        font=dict(color='#c9d1d9'),
        height=320,
        margin=dict(l=20, r=20, t=50, b=20),
        xaxis=dict(
            color='#4a6fa5',
            gridcolor='#1e2d42',
            tickfont=dict(family='Share Tech Mono', size=13)
        ),
        yaxis=dict(
            color='#4a6fa5',
            gridcolor='#1e2d42',
            tickfont=dict(family='Share Tech Mono'),
            title='Number of Frames'
        )
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Visualization 5 — Top Suspicious Frames Grid (video only)
# ─────────────────────────────────────────────────────────────

def show_suspicious_frames(frames, fake_scores, frame_numbers):
    """
    Displays a row of the top 5 frames with highest fake score.
    Caption under each: frame number + fake % + label.

    Args:
        frames        : list of PIL.Image — all sampled frames
        fake_scores   : list of float     — fake probability per frame
        frame_numbers : list of int       — original frame indices
    """
    indexed = sorted(
        zip(fake_scores, frame_numbers, frames),
        key=lambda x: x[0],
        reverse=True
    )
    top5 = indexed[:5]

    cols = st.columns(5)
    for i, (score, fnum, frame_img) in enumerate(top5):
        pct = score * 100
        is_fake = score >= 0.5
        with cols[i]:
            st.image(frame_img, use_container_width=True)
            color_class = "fake-caption" if is_fake else "real-caption"
            label_txt = "FAKE" if is_fake else "REAL"
            st.markdown(
                f'<div class="frame-caption {color_class}">'
                f'Frame {fnum}<br>{pct:.1f}% [{label_txt}]</div>',
                unsafe_allow_html=True
            )


# ─────────────────────────────────────────────────────────────
# Visualization 6 — Grad-CAM Heatmap Display
# ─────────────────────────────────────────────────────────────

def show_gradcam_display(image_pil, blended):
    """
    Displays original image alongside the Grad-CAM heatmap overlay.
    Includes a legend explaining warm vs cool zones.

    Args:
        image_pil : PIL.Image — original input image
        blended   : PIL.Image — heatmap blended onto original (from gradcam.py)
    """
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            '<div class="section-header">Original Image</div>',
            unsafe_allow_html=True
        )
        st.image(image_pil.convert("RGB"), use_container_width=True)

    with col2:
        st.markdown(
            '<div class="section-header">Grad-CAM Attention Map</div>',
            unsafe_allow_html=True
        )
        st.image(blended, use_container_width=True)

    st.markdown("""
    <div style="font-family:'Share Tech Mono',monospace; font-size:0.78rem;
                color:#4a6fa5; letter-spacing:1px; margin-top:0.5rem;">
    ⚡ Red/warm zones = regions Xception focused on most heavily for this decision.
    Blue/cool zones = low-attention areas.
    </div>
    """, unsafe_allow_html=True)
