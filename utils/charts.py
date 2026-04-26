import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ─────────────────────────────────────────────────────────────
# Verdict Card
# ─────────────────────────────────────────────────────────────

def show_verdict_card(label, fake_score):
    fake_pct = fake_score * 100
    is_fake  = label.lower() == "fake"

    verdict_text = "LIKELY FAKE" if is_fake else "LIKELY REAL"
    color        = "#ff3355" if is_fake else "#00ff88"
    bg           = "rgba(255,51,85,0.06)" if is_fake else "rgba(0,255,136,0.06)"
    border       = "#ff3355" if is_fake else "#00ff88"

    conf_pct = fake_pct if is_fake else (1 - fake_score) * 100
    frames_note = f"Analysis yields a manipulation probability of {fake_pct:.1f}%."
    auth_note   = "Significant artifacts detected. Likely manipulated." if is_fake else "No significant artifacts detected. This media appears authentic."

    st.markdown(f"""
    <div style="background:{bg}; border:1px solid {border}; border-radius:4px;
                padding:2rem 2.5rem; margin-bottom:1.5rem;">
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.75rem;
                    color:{color}; letter-spacing:3px; margin-bottom:0.8rem;">
            {'⚠' if is_fake else '✓'} {'MANIPULATION DETECTED' if is_fake else 'APPEARS AUTHENTIC'}
        </div>
        <div style="font-family:'Share Tech Mono',monospace; font-size:3.5rem;
                    color:{color}; letter-spacing:4px; line-height:1;
                    text-shadow: 0 0 30px {color}88; margin-bottom:0.8rem;">
            {fake_pct:.1f}%
        </div>
        <div style="font-family:'Share Tech Mono',monospace; font-size:1.2rem;
                    color:{color}; letter-spacing:3px; margin-bottom:1rem;">
            {verdict_text}
        </div>
        <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem;
                    color:#4a7a6a; line-height:1.6;">
            {frames_note}<br>{auth_note}
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Metric Cards Row
# ─────────────────────────────────────────────────────────────

def show_metric_cards(fake_score, total_frames, fake_count, uncertain_count):
    clean_count = total_frames - fake_count - uncertain_count
    fake_pct    = fake_score * 100

    metrics = [
        ("FAKE SCORE",      f"{fake_pct:.1f}%", "#00ff88" if fake_pct < 40 else "#ff3355"),
        ("FRAMES SCANNED",  str(total_frames),   "#ffffff"),
        ("FLAGGED FAKE",    str(fake_count),      "#ff3355"),
        ("UNCERTAIN",       str(uncertain_count), "#ffaa00"),
        ("CLEAN",           str(clean_count),     "#00ff88"),
    ]

    cols = st.columns(5)
    for col, (label, value, color) in zip(cols, metrics):
        with col:
            st.markdown(f"""
            <div style="border-left:2px solid {color}; padding:0.8rem 1rem;
                        background:rgba(255,255,255,0.02); border-radius:0 4px 4px 0;">
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.65rem;
                            color:#4a7a6a; letter-spacing:2px; margin-bottom:0.5rem;">
                    {label}
                </div>
                <div style="font-family:'Share Tech Mono',monospace; font-size:1.8rem;
                            color:{color}; font-weight:bold;">
                    {value}
                </div>
            </div>
            """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Analysis Charts — Gauge + Donut + Timeline (3 columns)
# ─────────────────────────────────────────────────────────────

def show_analysis_charts(fake_score, fake_scores, frame_numbers):
    fake_pct   = fake_score * 100
    fake_count = sum(1 for s in fake_scores if s >= 0.6)
    uncertain  = sum(1 for s in fake_scores if 0.4 <= s < 0.6)
    real_count = len(fake_scores) - fake_count - uncertain

    col1, col2, col3 = st.columns(3)

    # ── Gauge ──
    with col1:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.7rem; color:#4a7a6a; letter-spacing:2px; margin-bottom:0.5rem;">OVERALL SCORE</div>', unsafe_allow_html=True)
        gauge_color = "#ff3355" if fake_pct >= 50 else "#00ff88"
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=fake_pct,
            number={'suffix': '%', 'font': {'size': 28, 'color': gauge_color, 'family': 'Share Tech Mono'}},
            gauge={
                'axis': {'range': [0, 100], 'tickcolor': '#1a3a2a',
                         'tickfont': {'color': '#4a7a6a', 'family': 'Share Tech Mono', 'size': 10}},
                'bar': {'color': gauge_color, 'thickness': 0.3},
                'bgcolor': '#050a0f',
                'borderwidth': 0,
                'steps': [
                    {'range': [0,  40],  'color': 'rgba(0,255,136,0.08)'},
                    {'range': [40, 60],  'color': 'rgba(255,170,0,0.08)'},
                    {'range': [60, 100], 'color': 'rgba(255,51,85,0.08)'}
                ],
                'threshold': {'line': {'color': '#00e5ff', 'width': 2},
                              'thickness': 0.8, 'value': 50},
                'shape': 'angular'
            },
            title={'text': 'LIKELY REAL' if fake_pct < 50 else 'LIKELY FAKE',
                   'font': {'size': 11, 'color': gauge_color, 'family': 'Share Tech Mono'}}
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=220, margin=dict(l=20, r=20, t=20, b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── Donut ──
    with col2:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.7rem; color:#4a7a6a; letter-spacing:2px; margin-bottom:0.5rem;">FRAME BREAKDOWN</div>', unsafe_allow_html=True)
        fig2 = go.Figure(go.Pie(
            labels=['REAL', 'UNCERTAIN', 'FAKE'],
            values=[real_count, uncertain, fake_count],
            hole=0.65,
            marker=dict(colors=['#00ff88', '#ffaa00', '#ff3355'],
                        line=dict(color='#050a0f', width=2)),
            textfont=dict(family='Share Tech Mono', size=10, color='#ffffff'),
            hovertemplate='%{label}: %{value} frames<extra></extra>'
        ))
        fig2.add_annotation(
            text=f"<b>{len(fake_scores)}</b><br><span style='font-size:10px'>FRAMES</span>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(family='Share Tech Mono', size=16, color='#ffffff')
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            height=220, margin=dict(l=10, r=10, t=10, b=10),
            legend=dict(font=dict(family='Share Tech Mono', size=10, color='#4a7a6a'),
                        bgcolor='rgba(0,0,0,0)')
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── Timeline ──
    with col3:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.7rem; color:#4a7a6a; letter-spacing:2px; margin-bottom:0.5rem;">FRAME-BY-FRAME FAKE PROBABILITY</div>', unsafe_allow_html=True)
        fake_pcts  = [s * 100 for s in fake_scores]
        pt_colors  = ['#ff3355' if p >= 60 else '#ffaa00' if p >= 40 else '#00ff88' for p in fake_pcts]

        fig3 = go.Figure()
        fig3.add_hrect(y0=60, y1=100, fillcolor="rgba(255,51,85,0.05)",   line_width=0)
        fig3.add_hrect(y0=40, y1=60,  fillcolor="rgba(255,170,0,0.05)",   line_width=0)
        fig3.add_hrect(y0=0,  y1=40,  fillcolor="rgba(0,255,136,0.05)",   line_width=0)

        fig3.add_trace(go.Scatter(
            x=frame_numbers, y=fake_pcts,
            mode='lines+markers',
            line=dict(color='#00e5ff', width=1.5),
            marker=dict(color=pt_colors, size=6, line=dict(color='#050a0f', width=1)),
            hovertemplate='Frame %{x}<br>%{y:.1f}%<extra></extra>'
        ))
        fig3.add_hline(y=60, line_dash="dash", line_color="#ff3355",  line_width=1,
                       annotation_text="FAKE",      annotation_font=dict(color='#ff3355',  family='Share Tech Mono', size=9), annotation_position="right")
        fig3.add_hline(y=40, line_dash="dash", line_color="#ffaa00",  line_width=1,
                       annotation_text="UNCERTAIN", annotation_font=dict(color='#ffaa00',  family='Share Tech Mono', size=9), annotation_position="right")
        fig3.add_hline(y=0,  line_dash="dash", line_color="#00ff88",  line_width=1,
                       annotation_text="REAL",      annotation_font=dict(color='#00ff88',  family='Share Tech Mono', size=9), annotation_position="right")

        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,10,15,0.8)',
            height=220, margin=dict(l=10, r=50, t=10, b=30),
            xaxis=dict(color='#4a7a6a', gridcolor='#0d2018',
                       tickfont=dict(family='Share Tech Mono', size=9),
                       title=dict(text='FRAME', font=dict(family='Share Tech Mono', size=9, color='#4a7a6a'))),
            yaxis=dict(color='#4a7a6a', gridcolor='#0d2018', range=[0, 100],
                       tickfont=dict(family='Share Tech Mono', size=9),
                       title=dict(text='FAKE PROB', font=dict(family='Share Tech Mono', size=9, color='#4a7a6a'))),
            showlegend=False
        )
        st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Score Distribution Histogram
# ─────────────────────────────────────────────────────────────

def show_score_distribution(fake_scores):
    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=fake_scores, nbinsx=20,
        marker=dict(color='#00e5ff', line=dict(color='#050a0f', width=1)),
        hovertemplate='Score: %{x:.2f}<br>Count: %{y}<extra></extra>'
    ))
    fig.add_vline(x=0.4, line_dash="dash", line_color="#ffaa00", line_width=1.5,
                  annotation_text="", annotation_font=dict(color='#ffaa00'))
    fig.add_vline(x=0.6, line_dash="dash", line_color="#ff3355", line_width=1.5,
                  annotation_text="", annotation_font=dict(color='#ff3355'))

    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,10,15,0.8)',
        height=200, margin=dict(l=10, r=10, t=10, b=40),
        xaxis=dict(title=dict(text='FAKE SCORE (0=real, 1=fake)',
                              font=dict(family='Share Tech Mono', size=9, color='#4a7a6a')),
                   color='#4a7a6a', gridcolor='#0d2018',
                   tickfont=dict(family='Share Tech Mono', size=9)),
        yaxis=dict(title=dict(text='FRAME COUNT',
                              font=dict(family='Share Tech Mono', size=9, color='#4a7a6a')),
                   color='#4a7a6a', gridcolor='#0d2018',
                   tickfont=dict(family='Share Tech Mono', size=9))
    )
    st.plotly_chart(fig, use_container_width=True)


# ─────────────────────────────────────────────────────────────
# Grad-CAM Grid — all frames
# ─────────────────────────────────────────────────────────────

def show_gradcam_grid(frames, blended_list, frame_numbers, fake_scores, zone_scores_list):
    """
    Shows all frames in a scrollable grid.
    Each row: frame selector label, original + heatmap side by side.
    Below each pair: region suspicion scores for that frame.
    """
    for i, (orig, blended, fnum, fscore, zone_scores) in enumerate(
        zip(frames, blended_list, frame_numbers, fake_scores, zone_scores_list)
    ):
        pct      = fscore * 100
        is_fake  = fscore >= 0.6
        is_unc   = 0.4 <= fscore < 0.6
        color    = "#ff3355" if is_fake else "#ffaa00" if is_unc else "#00ff88"
        label    = "FAKE" if is_fake else "UNCERTAIN" if is_unc else "REAL"

        # Frame header
        st.markdown(f"""
        <div style="display:flex; align-items:center; gap:1rem; margin:1.5rem 0 0.5rem 0;">
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem;
                        color:#ffffff; background:#0d2018; border:1px solid #1a3a2a;
                        padding:0.2rem 0.8rem; border-radius:2px;">
                FRAME {fnum}
            </div>
            <div style="font-family:'Share Tech Mono',monospace; font-size:0.8rem;
                        color:{color}; background:rgba(0,0,0,0.3); border:1px solid {color}44;
                        padding:0.2rem 0.8rem; border-radius:2px;">
                SCORE {pct:.1f}% [{label}]
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Original + heatmap
        col1, col2 = st.columns(2)
        with col1:
            st.markdown('<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.65rem; color:#4a7a6a; letter-spacing:2px; margin-bottom:0.3rem;">ORIGINAL</div>', unsafe_allow_html=True)
            st.image(orig.convert("RGB"), use_container_width=True)
        with col2:
            st.markdown('<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.65rem; color:#4a7a6a; letter-spacing:2px; margin-bottom:0.3rem;">GRAD-CAM OVERLAY</div>', unsafe_allow_html=True)
            st.image(blended, use_container_width=True)

        # Region scores for this frame
        show_region_scores(zone_scores)

        # Divider
        st.markdown('<hr style="border:none; border-top:1px solid #0d2018; margin:1rem 0;">', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Region Suspicion Scores
# ─────────────────────────────────────────────────────────────

def show_region_scores(zone_scores):
    """
    Shows facial region activation scores:
    Left side — progress bars per zone
    Right side — horizontal bar chart
    """
    col_left, col_right = st.columns(2)

    zone_names = list(zone_scores.keys())
    zone_vals  = list(zone_scores.values())

    with col_left:
        st.markdown('<div style="font-family:\'Share Tech Mono\',monospace; font-size:0.65rem; color:#4a7a6a; letter-spacing:2px; margin-bottom:0.8rem;">GRAD-CAM ACTIVATION PER FACIAL ZONE</div>', unsafe_allow_html=True)
        for name, val in zone_scores.items():
            color = "#ff3355" if val >= 30 else "#ffaa00" if val >= 15 else "#00ff88"
            bar_w = min(val * 2, 100)
            st.markdown(f"""
            <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.6rem;">
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem;
                            color:#8ab8a8; width:80px; letter-spacing:1px;">
                    {name.upper()}
                </div>
                <div style="flex:1; background:#0d2018; border-radius:2px; height:4px;">
                    <div style="width:{bar_w}%; background:{color}; height:4px; border-radius:2px;"></div>
                </div>
                <div style="font-family:'Share Tech Mono',monospace; font-size:0.72rem;
                            color:{color}; width:45px; text-align:right;">
                    {val:.1f}%
                </div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        colors = ["#ff3355" if v >= 30 else "#ffaa00" if v >= 15 else "#00ff88" for v in zone_vals]
        fig = go.Figure(go.Bar(
            x=zone_vals,
            y=zone_names,
            orientation='h',
            marker=dict(color=colors, line=dict(color='#050a0f', width=1)),
            text=[f"{v:.1f}%" for v in zone_vals],
            textposition='outside',
            textfont=dict(family='Share Tech Mono', size=9, color='#ffffff'),
            hovertemplate='%{y}: %{x:.1f}%<extra></extra>'
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(5,10,15,0.8)',
            height=220, margin=dict(l=10, r=60, t=10, b=10),
            xaxis=dict(title=dict(text='GRAD-CAM ACTIVATION %',
                                  font=dict(family='Share Tech Mono', size=9, color='#4a7a6a')),
                       color='#4a7a6a', gridcolor='#0d2018', range=[0, 55],
                       tickfont=dict(family='Share Tech Mono', size=9)),
            yaxis=dict(color='#8ab8a8', tickfont=dict(family='Share Tech Mono', size=9),
                       categoryorder='array', categoryarray=list(reversed(zone_names)))
        )
        st.plotly_chart(fig, use_container_width=True)
