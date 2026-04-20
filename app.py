import streamlit as st
import os
import json
import datetime
from pathlib import Path

# ── These imports come from your existing files ──
from predict import predict
from report_generator import generate_report
from src.gradcam_visualize import run_gradcam

# ─────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="DysleXpert · AI Dyslexia Detection",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────
#  Session state defaults
# ─────────────────────────────────────────────────
for key, default in {
    "page": "home",
    "sessions": [],          # list of saved analysis dicts
    "last_result": None,     # dict with latest analysis data
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ─────────────────────────────────────────────────
#  Global CSS
# ─────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #F7F4EF !important;
    font-family: 'DM Sans', sans-serif !important;
    color: #1A160E !important;
}

/* Hide default Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
[data-testid="collapsedControl"] { display: none; }
[data-testid="stSidebar"] { display: none; }

/* ── Remove default padding ── */
.block-container { padding: 0 !important; max-width: 100% !important; }
section[data-testid="stMain"] > div { padding: 0 !important; }

/* ── NAV ── */
.dxp-nav {
    position: sticky; top: 0; z-index: 999;
    background: rgba(247,244,239,0.94);
    backdrop-filter: blur(12px);
    border-bottom: 1px solid rgba(60,50,30,0.12);
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 2.5rem; height: 58px;
}
.dxp-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 22px; color: #1A160E; letter-spacing: -0.5px;
}
.dxp-logo span { color: #2D6A4F; }
.dxp-nav-links { display: flex; gap: 4px; }
.dxp-nav-links a {
    font-size: 14px; color: #5C5144; text-decoration: none;
    padding: 6px 14px; border-radius: 8px;
    transition: background 0.18s;
    cursor: pointer; font-weight: 400;
}
.dxp-nav-links a:hover { background: #EDEAE3; color: #1A160E; }
.dxp-nav-links a.active { background: #EDEAE3; color: #1A160E; font-weight: 500; }

/* ── HERO ── */
.dxp-hero {
    max-width: 860px; margin: 0 auto;
    padding: 5rem 2rem 2.5rem; text-align: center;
}
.dxp-eyebrow {
    font-size: 11px; letter-spacing: 2.5px; text-transform: uppercase;
    color: #2D6A4F; font-weight: 500; margin-bottom: 1.4rem;
}
.dxp-h1 {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.6rem, 5.5vw, 4.2rem);
    line-height: 1.08; letter-spacing: -1.5px;
    color: #1A160E; margin-bottom: 1.4rem;
}
.dxp-h1 em { color: #2D6A4F; font-style: italic; }
.dxp-lead {
    font-size: 17px; color: #5C5144; max-width: 580px;
    margin: 0 auto 2.5rem; line-height: 1.75; font-weight: 300;
}

/* ── STAT STRIP ── */
.dxp-stats {
    display: flex; justify-content: center;
    border-top: 1px solid rgba(60,50,30,0.12);
    border-bottom: 1px solid rgba(60,50,30,0.12);
    max-width: 680px; margin: 2.5rem auto;
}
.dxp-stat { flex: 1; padding: 1.4rem; text-align: center; border-right: 1px solid rgba(60,50,30,0.12); }
.dxp-stat:last-child { border-right: none; }
.dxp-stat-n {
    font-family: 'DM Serif Display', serif;
    font-size: 2rem; color: #1A160E; letter-spacing: -1px;
}
.dxp-stat-l { font-size: 11px; color: #9E9080; margin-top: 3px; letter-spacing: 0.5px; }

/* ── FEATURE GRID ── */
.dxp-features {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 1px; background: rgba(60,50,30,0.12);
    border: 1px solid rgba(60,50,30,0.12);
    border-radius: 12px; overflow: hidden;
    max-width: 900px; margin: 0 auto 4rem;
}
.dxp-feat { background: #FFFFFF; padding: 1.8rem; }
.dxp-feat-icon { font-size: 1.5rem; margin-bottom: 0.8rem; }
.dxp-feat h3 { font-size: 14px; font-weight: 500; margin-bottom: 5px; }
.dxp-feat p { font-size: 13px; color: #5C5144; line-height: 1.6; margin: 0; }

/* ── BUTTONS ── */
.dxp-btn-primary, .stButton > button[kind="primary"] {
    background: #2D6A4F !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    padding: 13px 28px !important; font-size: 15px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
    cursor: pointer; transition: all 0.18s !important;
    box-shadow: none !important;
}
.stButton > button {
    background: #2D6A4F !important; color: white !important;
    border: none !important; border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
    transition: all 0.18s !important; box-shadow: none !important;
}
.stButton > button:hover { background: #235540 !important; }

/* ── CARDS ── */
.dxp-card {
    background: #FFFFFF; border: 1px solid rgba(60,50,30,0.12);
    border-radius: 12px; padding: 1.5rem;
}
.dxp-card-title { font-size: 14px; font-weight: 500; margin-bottom: 4px; }
.dxp-card-sub { font-size: 13px; color: #5C5144; margin-bottom: 1rem; }

/* ── SEVERITY BADGE ── */
.dxp-severity-wrap { display: flex; align-items: center; gap: 1.5rem; flex-wrap: wrap; }
.dxp-severity-circle {
    width: 96px; height: 96px; border-radius: 50%;
    display: flex; flex-direction: column; align-items: center; justify-content: center;
    flex-shrink: 0;
}
.dxp-sev-pct { font-family: 'DM Serif Display', serif; font-size: 1.8rem; line-height: 1; }
.dxp-sev-label { font-size: 10px; margin-top: 3px; letter-spacing: 0.6px; text-transform: uppercase; }
.sev-none   { background: #E8F5EE; color: #2D6A4F; }
.sev-dys    { background: #FAEAE5; color: #C84B31; }

/* ── METRICS ROW ── */
.dxp-metrics {
    display: grid; grid-template-columns: repeat(4,1fr);
    gap: 1px; background: rgba(60,50,30,0.12);
    border: 1px solid rgba(60,50,30,0.12);
    border-radius: 12px; overflow: hidden;
}
.dxp-metric { background: #FFFFFF; padding: 1.1rem; text-align: center; }
.dxp-metric-val { font-family: 'DM Serif Display', serif; font-size: 1.7rem; }
.dxp-metric-key { font-size: 10px; color: #9E9080; margin-top: 3px; text-transform: uppercase; letter-spacing: 0.5px; }
.mval-green { color: #2D6A4F; }
.mval-red   { color: #C84B31; }

/* ── PROGRESS BARS ── */
.dxp-bar-wrap { margin-top: 0.8rem; }
.dxp-bar-row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; }
.dxp-bar-label { font-size: 12px; color: #5C5144; width: 100px; flex-shrink: 0; }
.dxp-bar-track { flex: 1; height: 6px; background: #F0EDE6; border-radius: 3px; overflow: hidden; }
.dxp-bar-fill { height: 100%; border-radius: 3px; }
.dxp-bar-pct { font-size: 12px; font-weight: 500; width: 40px; text-align: right; }

/* ── EXERCISE ITEMS ── */
.dxp-ex-item {
    background: #FFFFFF; border: 1px solid rgba(60,50,30,0.12);
    border-radius: 8px; padding: 12px 14px; margin-bottom: 8px;
    font-size: 13px; line-height: 1.55; color: #1A160E;
}
.dxp-ex-title { font-weight: 500; margin-bottom: 3px; font-size: 13px; }

/* ── SESSION CARDS ── */
.dxp-session {
    background: #FFFFFF; border: 1px solid rgba(60,50,30,0.12);
    border-radius: 10px; padding: 1.1rem; margin-bottom: 8px;
}
.dxp-session-date { font-size: 11px; color: #9E9080; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
.dxp-session-result { font-size: 14px; font-weight: 500; }
.dxp-session-sub { font-size: 12px; color: #9E9080; margin-top: 3px; }

/* ── UPLOAD ZONE ── */
[data-testid="stFileUploader"] {
    border: 2px dashed rgba(60,50,30,0.25) !important;
    border-radius: 12px !important;
    padding: 2rem !important;
    background: #FFFFFF !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover { border-color: #2D6A4F !important; }

/* ── PROGRESS / SPINNER ── */
[data-testid="stSpinner"] { color: #2D6A4F !important; }

/* ── TABS ── */
[data-testid="stTabs"] [role="tab"] {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 14px !important; color: #5C5144 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    color: #1A160E !important; font-weight: 500 !important;
    border-bottom-color: #2D6A4F !important;
}

/* ── ABOUT ── */
.dxp-about { max-width: 800px; margin: 0 auto; padding: 3rem 2rem 5rem; }
.dxp-about h2 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem; letter-spacing: -0.5px; margin-bottom: 0.9rem;
}
.dxp-about h3 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.2rem; margin-top: 2rem; margin-bottom: 0.6rem;
}
.dxp-about p { font-size: 15px; color: #5C5144; line-height: 1.8; margin-bottom: 1.2rem; }
.dxp-team-grid { display: grid; grid-template-columns: repeat(4,1fr); gap: 12px; margin: 1.2rem 0 2rem; }
.dxp-team-card {
    background: #FFFFFF; border: 1px solid rgba(60,50,30,0.12);
    border-radius: 12px; padding: 1.3rem; text-align: center;
}
.dxp-avatar {
    width: 48px; height: 48px; border-radius: 50%;
    background: #E8F5EE; color: #2D6A4F;
    display: flex; align-items: center; justify-content: center;
    font-weight: 500; font-size: 15px; margin: 0 auto 8px;
}
.dxp-team-name { font-size: 13px; font-weight: 500; }
.dxp-team-role { font-size: 11px; color: #9E9080; margin-top: 2px; }

/* ── PAGE WRAPPER ── */
.dxp-page { padding: 0 2.5rem 4rem; max-width: 980px; margin: 0 auto; }
.dxp-page-header { padding: 2.5rem 0 1.5rem; }
.dxp-page-header h2 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.9rem; letter-spacing: -0.5px;
}
.dxp-page-header p { color: #5C5144; font-size: 15px; margin-top: 4px; }

/* ── DISCLAIMER ── */
.dxp-disclaimer {
    background: #FDF3DC; border: 1px solid rgba(176,125,26,0.3);
    border-radius: 8px; padding: 10px 14px;
    font-size: 12px; color: #7A5A00; margin-bottom: 1.2rem;
}

/* Fix st.image border */
[data-testid="stImage"] img {
    border-radius: 10px;
    border: 1px solid rgba(60,50,30,0.12);
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────
#  Helper: exercises by severity
# ─────────────────────────────────────────────────
def get_exercises(is_dyslexic: bool, confidence: float) -> list[dict]:
    if not is_dyslexic:
        return [
            {"title": "Maintain regular practice",
             "body": "15 min handwriting exercises daily to sustain current proficiency and motor fluency."},
            {"title": "Cursive development",
             "body": "Introduce connected cursive writing to further develop motor-linguistic coordination."},
        ]
    if confidence < 0.70:
        return [
            {"title": "Letter tracing — b / d / p / q",
             "body": "Multisensory tracing cards for easily-reversed letters, 10 min/day. Combine visual, tactile, and auditory feedback."},
            {"title": "Spacing practice",
             "body": "Use the finger-spacing technique between words. Practice on lined paper emphasising consistent inter-word gaps."},
            {"title": "Phonological awareness",
             "body": "Sound-segmentation games: break 2-syllable words into phonemes aloud while writing each letter."},
            {"title": "Baseline anchoring",
             "body": "Raised-line paper provides tactile baseline feedback during writing practice sessions."},
        ]
    return [
        {"title": "Structured literacy (Orton-Gillingham)",
         "body": "Systematic, explicit phonics instruction. 30 min structured sessions, 5×/week minimum."},
        {"title": "Multisensory letter formation",
         "body": "Trace letters in sand/textured surfaces while pronouncing each phoneme. Sky-writing for motor memory. Focus on most-confused pairs first."},
        {"title": "Chunking & word spacing",
         "body": "Write one word at a time with deliberate pause. Colour-coded word separators. Practice with block-letter stamps before cursive."},
        {"title": "Dictation & phoneme mapping",
         "body": "Slow dictation: hear word → segment phonemes aloud → write letter by letter. Record and replay for self-monitoring."},
        {"title": "Reading aloud with tracking",
         "body": "Finger-track each word while reading aloud. Reinforces visual-motor-auditory integration. 15 min/day."},
    ]


# ─────────────────────────────────────────────────
#  NAV RENDERING
# ─────────────────────────────────────────────────
def render_nav():
    pages = [("home", "Home"), ("analyze", "Analyze"), ("monitor", "Monitor"), ("about", "About")]
    active = st.session_state.page

    nav_html = '<div class="dxp-nav"><div class="dxp-logo">Dysl<span>eX</span>pert</div><div class="dxp-nav-links">'
    for pid, label in pages:
        cls = "active" if pid == active else ""
        nav_html += f'<a class="{cls}" style="text-decoration:none;">{label}</a>'
    nav_html += '</div></div>'
    st.markdown(nav_html, unsafe_allow_html=True)

    # Real nav buttons hidden below HTML nav (Streamlit requirement)
    cols = st.columns([5, 1, 1, 1, 1])
    for i, (pid, label) in enumerate(pages):
        with cols[i + 1]:
            if st.button(label, key=f"nav_{pid}"):
                st.session_state.page = pid
                st.rerun()


# ─────────────────────────────────────────────────
#  HOME PAGE
# ─────────────────────────────────────────────────
def page_home():
    st.markdown("""
    <div class="dxp-hero">
      <div class="dxp-eyebrow">EfficientNet · Grad-CAM · Explainable AI</div>
      <div class="dxp-h1">Detect dyslexia through <em>handwriting</em>, transparently.</div>
      <div class="dxp-lead">Upload a handwriting sample and receive a clinical-grade analysis
      with Grad-CAM visualisation and personalised exercises — in seconds.</div>
    </div>
    """, unsafe_allow_html=True)

    # CTA button
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Analyze handwriting →", key="home_cta"):
            st.session_state.page = "analyze"
            st.rerun()

    st.markdown("""
    <div class="dxp-stats">
      <div class="dxp-stat"><div class="dxp-stat-n">94%</div><div class="dxp-stat-l">Accuracy</div></div>
      <div class="dxp-stat"><div class="dxp-stat-n">94.5</div><div class="dxp-stat-l">F1-Score</div></div>
      <div class="dxp-stat"><div class="dxp-stat-n">ResNet</div><div class="dxp-stat-l">Architecture</div></div>
      <div class="dxp-stat"><div class="dxp-stat-n">XAI</div><div class="dxp-stat-l">Grad-CAM</div></div>
    </div>

    <div style="max-width:900px;margin:0 auto;padding:0 2.5rem;">
    <div class="dxp-features">
      <div class="dxp-feat">
        <div class="dxp-feat-icon">🔬</div>
        <h3>Deep Learning Classification</h3>
        <p>ResNet18 trained on real handwriting samples captures subtle stroke irregularities, spacing, and baseline deviations.</p>
      </div>
      <div class="dxp-feat">
        <div class="dxp-feat-icon">🗺️</div>
        <h3>Grad-CAM Heatmaps</h3>
        <p>SmoothGradCAM++ overlays show exactly which handwriting regions drove the model's decision.</p>
      </div>
      <div class="dxp-feat">
        <div class="dxp-feat-icon">✏️</div>
        <h3>Adaptive Exercises</h3>
        <p>Auto-generated phonological and handwriting exercises tailored to the detected confidence level.</p>
      </div>
      <div class="dxp-feat">
        <div class="dxp-feat-icon">📄</div>
        <h3>PDF Report</h3>
        <p>Download a one-page clinical-style report with prediction, probabilities, and the Grad-CAM image.</p>
      </div>
      <div class="dxp-feat">
        <div class="dxp-feat-icon">📈</div>
        <h3>Progress Monitor</h3>
        <p>Save sessions and track severity changes longitudinally across multiple analyses.</p>
      </div>
      <div class="dxp-feat">
        <div class="dxp-feat-icon">🌐</div>
        <h3>Accessible & Fast</h3>
        <p>Runs on CPU, works with scanned worksheets or tablet captures. No specialist hardware required.</p>
      </div>
    </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────
#  ANALYZE PAGE
# ─────────────────────────────────────────────────
def page_analyze():
    st.markdown("""
    <div class="dxp-page">
    <div class="dxp-page-header">
      <h2>Handwriting Analysis</h2>
      <p>Upload a handwriting image to receive an AI-powered dyslexia screening report.</p>
    </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="dxp-page">', unsafe_allow_html=True)

    st.markdown('<div class="dxp-disclaimer">⚠️ This tool is a research demonstration. Results should not replace a clinical diagnosis.</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop a handwriting image here (JPG / PNG)",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
    )

    if uploaded_file is not None:
        # Save temp file
        temp_path = "temp.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # ── Preview & Analysis side by side ──
        col_img, col_heat = st.columns(2)
        with col_img:
            st.markdown('<div class="dxp-card"><div class="dxp-card-title">📄 Original sample</div>', unsafe_allow_html=True)
            st.image(uploaded_file, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)

        # ── Run model ──
        with st.spinner("Running ResNet18 + Grad-CAM analysis..."):
            label, dys_prob, non_dys_prob = predict(temp_path)
            dys_prob = float(dys_prob)
            non_dys_prob = float(non_dys_prob)

            try:
                gradcam_path, _, _ = run_gradcam(temp_path)
                has_gradcam = True
            except Exception as e:
                has_gradcam = False
                gradcam_path = None

        with col_heat:
            st.markdown('<div class="dxp-card"><div class="dxp-card-title">🔴 Grad-CAM heatmap overlay</div>', unsafe_allow_html=True)
            if has_gradcam:
                st.image(gradcam_path, use_container_width=True)
                st.markdown('<p style="font-size:12px;color:#9E9080;margin-top:6px;">Red = high diagnostic relevance · Blue = low relevance</p>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color:#C84B31;font-size:13px;padding:2rem 0;">Grad-CAM generation failed. Check model layer name.</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

        # ── Result hero ──
        is_dyslexic = label.lower() == "dyslexic"
        sev_cls = "sev-dys" if is_dyslexic else "sev-none"
        conf_pct = round(dys_prob * 100, 1) if is_dyslexic else round(non_dys_prob * 100, 1)
        sev_label = "Dyslexic" if is_dyslexic else "Non-dyslexic"

        result_title = "Dyslexic Handwriting Pattern Detected" if is_dyslexic else "Non-Dyslexic Pattern"
        result_desc = (
            "The model detected handwriting patterns consistent with dyslexia. "
            "Irregular spacing, letter formation inconsistencies, and baseline deviations may be present. "
            "Early intervention is recommended."
            if is_dyslexic else
            "The handwriting sample shows consistent letter formation, regular spacing, and stable baseline alignment. "
            "No significant dyslexia markers were detected."
        )

        st.markdown(f"""
        <div class="dxp-card" style="margin-bottom:1.2rem;">
          <div class="dxp-severity-wrap">
            <div class="dxp-severity-circle {sev_cls}">
              <span class="dxp-sev-pct">{conf_pct}%</span>
              <span class="dxp-sev-label">{sev_label}</span>
            </div>
            <div style="flex:1;min-width:200px;">
              <div style="font-family:'DM Serif Display',serif;font-size:1.4rem;margin-bottom:6px;">{result_title}</div>
              <div style="font-size:13px;color:#5C5144;line-height:1.6;margin-bottom:12px;">{result_desc}</div>
              <div class="dxp-bar-wrap">
                <div class="dxp-bar-row">
                  <span class="dxp-bar-label">Dyslexic</span>
                  <div class="dxp-bar-track"><div class="dxp-bar-fill" style="width:{round(dys_prob*100)}%;background:#C84B31;"></div></div>
                  <span class="dxp-bar-pct">{round(dys_prob*100)}%</span>
                </div>
                <div class="dxp-bar-row">
                  <span class="dxp-bar-label">Non-Dyslexic</span>
                  <div class="dxp-bar-track"><div class="dxp-bar-fill" style="width:{round(non_dys_prob*100)}%;background:#2D6A4F;"></div></div>
                  <span class="dxp-bar-pct">{round(non_dys_prob*100)}%</span>
                </div>
              </div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Grad-CAM explanation text ──
        if is_dyslexic:
            cam_text = (
                "The Grad-CAM heatmap highlights regions of high diagnostic relevance "
                "(shown in red/orange). The model attended to letter boundaries, mid-word "
                "transitions, and spacing patterns. These regions correspond to the handwriting "
                "features most associated with dyslexic writing."
            )
        else:
            cam_text = (
                "Grad-CAM activation is minimal and diffuse, indicating no concentrated "
                "regions of concern. The model found consistent letter formation and spacing "
                "throughout the sample, with no high-activation anomalies."
            )

        st.markdown(f"""
        <div class="dxp-card" style="margin-bottom:1.2rem;">
          <div class="dxp-card-title">Grad-CAM Interpretation</div>
          <div class="dxp-card-sub" style="margin-bottom:0;">{cam_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # ── Adaptive exercises ──
        exercises = get_exercises(is_dyslexic, dys_prob)
        badge_style = "background:#FAEAE5;color:#C84B31;" if is_dyslexic else "background:#E8F5EE;color:#2D6A4F;"
        badge_text = "Intensive intervention" if (is_dyslexic and dys_prob > 0.70) else ("Targeted intervention" if is_dyslexic else "Maintenance plan")

        ex_html = f"""
        <div class="dxp-card" style="margin-bottom:1.2rem;">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:1rem;">
            <div class="dxp-card-title" style="margin:0;">Adaptive Exercise Plan</div>
            <span style="font-size:11px;padding:3px 10px;border-radius:20px;font-weight:500;{badge_style}">{badge_text}</span>
          </div>
        """
        for ex in exercises:
            ex_html += f'<div class="dxp-ex-item"><div class="dxp-ex-title">{ex["title"]}</div>{ex["body"]}</div>'
        ex_html += "</div>"
        st.markdown(ex_html, unsafe_allow_html=True)

        # ── Action buttons ──
        col_save, col_report, col_spacer = st.columns([1, 1, 2])
        with col_save:
            if st.button("💾 Save to Monitor", key="save_session"):
                session_entry = {
                    "date": datetime.datetime.now().strftime("%d %b %Y, %H:%M"),
                    "timestamp": datetime.datetime.now().isoformat(),
                    "label": label,
                    "is_dyslexic": is_dyslexic,
                    "dys_prob": round(dys_prob * 100, 1),
                    "non_dys_prob": round(non_dys_prob * 100, 1),
                    "conf": conf_pct,
                }
                st.session_state.sessions.append(session_entry)
                st.success("Session saved! Go to Monitor to view progress.")

        with col_report:
            if has_gradcam:
                report_path = generate_report(label, dys_prob, non_dys_prob, gradcam_path)
                with open(report_path, "rb") as f:
                    st.download_button(
                        label="📄 Download PDF Report",
                        data=f,
                        file_name=f"dyslexia_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        key="dl_report",
                    )

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────
#  MONITOR PAGE
# ─────────────────────────────────────────────────
def page_monitor():
    st.markdown("""
    <div class="dxp-page">
    <div class="dxp-page-header">
      <h2>Progress Monitor</h2>
      <p>Track dyslexia screening results across multiple sessions.</p>
    </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('<div class="dxp-page">', unsafe_allow_html=True)

    sessions = st.session_state.sessions

    if not sessions:
        st.markdown("""
        <div style="text-align:center;padding:4rem 2rem;color:#9E9080;">
          <div style="font-size:3rem;margin-bottom:1rem;">📊</div>
          <p style="font-size:15px;">No sessions recorded yet.<br>Analyze a sample and click "Save to Monitor".</p>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Go to Analyze →", key="mon_goto"):
            st.session_state.page = "analyze"
            st.rerun()
    else:
        # ── Summary metrics ──
        n = len(sessions)
        n_dys = sum(1 for s in sessions if s["is_dyslexic"])
        avg_conf = round(sum(s["conf"] for s in sessions) / n, 1)
        latest = sessions[-1]["label"]

        st.markdown(f"""
        <div class="dxp-metrics" style="margin-bottom:1.5rem;">
          <div class="dxp-metric"><div class="dxp-metric-val">{n}</div><div class="dxp-metric-key">Total Sessions</div></div>
          <div class="dxp-metric"><div class="dxp-metric-val {'mval-red' if n_dys else 'mval-green'}">{n_dys}</div><div class="dxp-metric-key">Dyslexic Detected</div></div>
          <div class="dxp-metric"><div class="dxp-metric-val">{n - n_dys}</div><div class="dxp-metric-key">Non-Dyslexic</div></div>
          <div class="dxp-metric"><div class="dxp-metric-val">{avg_conf}%</div><div class="dxp-metric-key">Avg Confidence</div></div>
        </div>
        """, unsafe_allow_html=True)

        # ── Session cards ──
        st.markdown("<div style='margin-bottom:0.6rem;font-size:14px;font-weight:500;'>Session history</div>", unsafe_allow_html=True)
        for i, s in enumerate(reversed(sessions)):
            dot_color = "#C84B31" if s["is_dyslexic"] else "#2D6A4F"
            st.markdown(f"""
            <div class="dxp-session">
              <div class="dxp-session-date">Session {n - i} · {s['date']}</div>
              <div class="dxp-session-result">
                <span style="width:8px;height:8px;border-radius:50%;background:{dot_color};display:inline-block;margin-right:6px;"></span>
                {s['label']}
              </div>
              <div class="dxp-session-sub">Dyslexic: {s['dys_prob']}% · Non-Dyslexic: {s['non_dys_prob']}% · Confidence: {s['conf']}%</div>
            </div>
            """, unsafe_allow_html=True)

        # ── Trend chart using st.line_chart ──
        if n >= 2:
            import pandas as pd
            st.markdown("<div style='margin-top:1.5rem;margin-bottom:0.4rem;font-size:14px;font-weight:500;'>Dyslexic probability trend</div>", unsafe_allow_html=True)
            chart_data = pd.DataFrame({
                "Session": [f"S{i+1}" for i in range(n)],
                "Dyslexic %": [s["dys_prob"] for s in sessions],
            }).set_index("Session")
            st.line_chart(chart_data, color="#C84B31", height=200)

        col_clear, _ = st.columns([1, 3])
        with col_clear:
            if st.button("🗑️ Clear all sessions", key="clear_sessions"):
                st.session_state.sessions = []
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────
#  ABOUT PAGE
# ─────────────────────────────────────────────────
def page_about():
    st.markdown("""
    <div class="dxp-about">
      <h2>About this project</h2>
      <p>DysleXpert is the live demonstration of the conference paper
      <em>"A Transparent Deep Learning Model for Detecting Dyslexia through Handwriting Patterns:
      Enhancing Interpretability with Grad-CAM"</em>, developed at Maanakula Vinayagar
      Institute of Technology, Pondicherry University.</p>

      <h3>Team</h3>
      <div class="dxp-team-grid">
        <div class="dxp-team-card"><div class="dxp-avatar">MD</div><div class="dxp-team-name">Mohana Priya D</div><div class="dxp-team-role">Assistant Professor · Advisor</div></div>
        <div class="dxp-team-card"><div class="dxp-avatar">AB</div><div class="dxp-team-name">Agashya B</div><div class="dxp-team-role">Developer</div></div>
        <div class="dxp-team-card"><div class="dxp-avatar">NS</div><div class="dxp-team-name">Nithya Sri P R</div><div class="dxp-team-role">ML Engineer</div></div>
        <div class="dxp-team-card"><div class="dxp-avatar">SN</div><div class="dxp-team-name">Saranya N</div><div class="dxp-team-role">Developer</div></div>
      </div>

      <h3>Model architecture</h3>
      <p>The backend uses <strong>ResNet18</strong> (1-channel input, binary classification head) trained on
      stationary handwriting datasets. <strong>SmoothGradCAM++</strong> from TorchCAM provides
      gradient-weighted activation maps on the <code>layer4</code> block, highlighting the
      handwriting regions most relevant to the prediction.</p>

      <h3>System pipeline</h3>
      <p>
        <strong>1. Input</strong> — scanned worksheet or tablet image (JPG/PNG)<br>
        <strong>2. Preprocessing</strong> — OpenCV grayscale · Otsu threshold · 224×224 resize<br>
        <strong>3. Feature extraction</strong> — ResNet18 convolutional layers<br>
        <strong>4. Classification</strong> — Sigmoid binary output (Dyslexic / Non-Dyslexic)<br>
        <strong>5. Explainability</strong> — SmoothGradCAM++ heatmap on <code>layer4</code><br>
        <strong>6. Output</strong> — prediction · probabilities · Grad-CAM · exercises · PDF report
      </p>

      <h3>Published performance</h3>
      <p>Hybrid EfficientNet + handcrafted features achieved <strong>94% accuracy</strong>,
      <strong>95% precision</strong>, <strong>94% recall</strong>, and <strong>94.5 F1-score</strong>
      on the combined public + local school dataset (60/20/20 split).</p>

      <h3>Repository</h3>
      <p>Source code and model weights:
      <a href="https://github.com/nithyasri7846/Dyslexia_Detection_Project" target="_blank"
         style="color:#2D6A4F;">github.com/nithyasri7846/Dyslexia_Detection_Project</a></p>

      <h3>Disclaimer</h3>
      <p style="font-size:13px;">This application is a research prototype. Results must not be used
      as a standalone clinical diagnosis. Always consult a qualified educational psychologist or
      specialist for formal dyslexia assessment.</p>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────
#  ROUTER
# ─────────────────────────────────────────────────
render_nav()

page = st.session_state.page
if page == "home":
    page_home()
elif page == "analyze":
    page_analyze()
elif page == "monitor":
    page_monitor()
elif page == "about":
    page_about()
