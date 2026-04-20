import streamlit as st
import os

from predict import predict
from report_generator import generate_report
from src.gradcam_visualize import run_gradcam

st.set_page_config(page_title="Dyslexia Detection", layout="wide")

# -------------------------
# Custom CSS
# -------------------------
st.markdown("""
<style>


.nav-btn .stButton > button {
    padding: 5px 18px !important;
    font-size: 14px !important;
    border-radius: 20px !important;
}

body {
    background: linear-gradient(135deg, #141E30, #243B55);
}

h1 {
    text-align: center;
    color: white;
    font-size: 45px;
}
.navbar {
    display: flex;
    justify-content: flex-end;
    gap: 40px;
    font-size: 18px;
    margin-bottom: 20px;
}

.nav-item {
    color: white;
    cursor: pointer;
    font-weight: 500;
}

.nav-item:hover {
    color: #00f5d4;
}

.about-box {
    text-align: center;
    margin-top: 0px;
    color: white;
}

.get-started {
    margin-top: 30px;
}

.stButton>button {
    background: linear-gradient(45deg, #00f5d4, #00bbf9);
    color: black;
    border-radius: 30px;
    padding: 8px 25px;
    font-weight: bold;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    box-shadow: 0px 0px 20px #00f5d4;
}

</style>
""", unsafe_allow_html=True)

# -------------------------
# Navigation State
# -------------------------
if "page" not in st.session_state:
    st.session_state.page = "home"

if "analysis_count" not in st.session_state:
    st.session_state.analysis_count = 0

# -------------------------
# Navbar
# -------------------------
col1, col2, col3, col4 = st.columns([5,1.2,1.2,0.8])

with col2:
    st.markdown("<div class='nav-btn'>", unsafe_allow_html=True)
    if st.button("Home"):
        st.session_state.page = "home"
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown("<div class='nav-btn'>", unsafe_allow_html=True)
    if st.button("Test"):
        st.session_state.page = "test"
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------
# HOME PAGE
# -------------------------
# -------------------------
# HOME PAGE
# -------------------------
# -------------------------
# HOME PAGE
# -------------------------
if st.session_state.page == "home":

    st.markdown("<h1>About Dyslexia</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class="about-box">
    <p>Dyslexia is a learning difficulty that affects reading and writing.</p>
    <p>It impacts the way the brain processes written and spoken language.</p>
    <p>It is not related to intelligence or lack of effort.</p>
    <p>Early detection can greatly improve learning outcomes.</p>
    <p>AI can assist in identifying patterns linked to dyslexia.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2,1,2])

    with col2:
        if st.button("Get Started"):
            st.session_state.page = "test"

# -------------------------
# TEST PAGE
# -------------------------
elif st.session_state.page == "test":

    st.title("📝 Dyslexia Test")

    uploaded_file = st.file_uploader(
        "Upload handwriting image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        temp_path = "temp.jpg"

        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        st.image(uploaded_file, width=600)

        with st.spinner("Analyzing..."):
            label, dys_prob, non_dys_prob = predict(temp_path)

        dys_prob = float(dys_prob)
        non_dys_prob = float(non_dys_prob)

        st.subheader(f"Prediction: {label}")

        st.write("### Probabilities")

        st.progress(dys_prob)
        st.write(f"Dyslexic: {round(dys_prob * 100, 2)}%")

        st.progress(non_dys_prob)
        st.write(f"Non-Dyslexic: {round(non_dys_prob * 100, 2)}%")

        st.session_state.analysis_count += 1

        if st.button("Generate Report"):

            gradcam_path, _, _ = run_gradcam(temp_path)

            report_path = generate_report(
                label,
                dys_prob,
                non_dys_prob,
                gradcam_path
            )

            with open(report_path, "rb") as file:
                st.download_button(
                    label="Download Report",
                    data=file,
                    file_name="dyslexia_report.pdf",
                    mime="application/pdf"
                )

'''import streamlit as st
import os
import datetime
import json

from predict import predict
from report_generator import generate_report
from src.gradcam_visualize import run_gradcam

# ─────────────────────────────────────────────────
st.set_page_config(
    page_title="DysleXpert",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────
for k, v in {
    "page": "home",
    "sessions": [],
    "profile": {
        "name": "",
        "age": "",
        "grade": "",
        "school": "",
        "therapist": "",
        "notes": "",
    },
}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ─────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&family=Sora:wght@400;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"],
[data-testid="stMain"], .main, section[data-testid="stMain"] > div {
    background: #0D0D0F !important;
    font-family: 'Inter', sans-serif !important;
    color: #E8E6E1 !important;
}

#MainMenu, footer, header,
[data-testid="collapsedControl"],
[data-testid="stSidebar"] { display: none !important; visibility: hidden !important; }

.block-container { padding: 0 !important; max-width: 100% !important; }

/* ─── NAV ─── */
.nav {
    display: flex; align-items: center; justify-content: space-between;
    padding: 0 2.5rem; height: 62px;
    background: rgba(13,13,15,0.95);
    border-bottom: 1px solid rgba(255,255,255,0.07);
    position: sticky; top: 0; z-index: 999;
}
.nav-logo {
    font-family: 'Sora', sans-serif; font-size: 20px;
    font-weight: 700; color: #FFFFFF; letter-spacing: -0.5px;
}
.nav-logo span { color: #4ADE80; }
.nav-links { display: flex; gap: 4px; }
.nav-link {
    font-size: 13px; font-weight: 500;
    color: rgba(232,230,225,0.55);
    padding: 7px 16px; border-radius: 8px;
    cursor: pointer; text-decoration: none;
    transition: all 0.15s;
    border: 1px solid transparent;
}
.nav-link:hover { color: #E8E6E1; background: rgba(255,255,255,0.06); }
.nav-link.active {
    color: #E8E6E1;
    background: rgba(255,255,255,0.08);
    border-color: rgba(255,255,255,0.1);
}

/* ─── BUTTONS ─── */
.stButton > button {
    background: #4ADE80 !important;
    color: #0D0D0F !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    transition: all 0.18s !important;
    box-shadow: none !important;
    cursor: pointer !important;
}
.stButton > button:hover {
    background: #22c55e !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ghost button variant */
.ghost-btn .stButton > button {
    background: transparent !important;
    color: #E8E6E1 !important;
    border: 1px solid rgba(255,255,255,0.18) !important;
}
.ghost-btn .stButton > button:hover {
    background: rgba(255,255,255,0.06) !important;
    border-color: rgba(255,255,255,0.3) !important;
}

/* ─── HOME ─── */
.home-wrap {
    min-height: calc(100vh - 62px);
    display: flex; flex-direction: column;
    align-items: center; justify-content: center;
    padding: 4rem 2rem 5rem;
    text-align: center;
}
.home-chip {
    display: inline-block;
    background: rgba(74,222,128,0.12);
    color: #4ADE80;
    border: 1px solid rgba(74,222,128,0.25);
    border-radius: 20px;
    font-size: 11px; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase;
    padding: 5px 14px; margin-bottom: 2rem;
}
.home-h1 {
    font-family: 'Sora', sans-serif;
    font-size: clamp(2.8rem, 6vw, 5rem);
    font-weight: 700; line-height: 1.06;
    letter-spacing: -2px; color: #FFFFFF;
    margin-bottom: 1.4rem;
}
.home-h1 em { color: #4ADE80; font-style: normal; }
.home-sub {
    font-size: 17px; font-weight: 300;
    color: rgba(232,230,225,0.6);
    max-width: 560px; margin: 0 auto 3rem;
    line-height: 1.75;
}

.home-cards {
    display: grid; grid-template-columns: repeat(3,1fr);
    gap: 1px; background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 16px; overflow: hidden;
    max-width: 820px; width: 100%; margin: 3.5rem auto 0;
    text-align: left;
}
.home-card { background: #111113; padding: 1.6rem; }
.home-card-icon { font-size: 1.4rem; margin-bottom: 0.8rem; }
.home-card h3 { font-size: 13px; font-weight: 600; color: #FFFFFF; margin-bottom: 5px; }
.home-card p { font-size: 12px; color: rgba(232,230,225,0.5); line-height: 1.6; }

/* ─── PAGE LAYOUT ─── */
.page-wrap { max-width: 980px; margin: 0 auto; padding: 2.5rem 2.5rem 5rem; }
.page-title {
    font-family: 'Sora', sans-serif;
    font-size: 1.7rem; font-weight: 700;
    color: #FFFFFF; letter-spacing: -0.5px;
    margin-bottom: 4px;
}
.page-sub { font-size: 14px; color: rgba(232,230,225,0.5); margin-bottom: 2rem; }

/* ─── UPLOAD ZONE ─── */
[data-testid="stFileUploader"] {
    background: #111113 !important;
    border: 2px dashed rgba(255,255,255,0.12) !important;
    border-radius: 14px !important;
    padding: 2rem !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: rgba(74,222,128,0.4) !important;
}
[data-testid="stFileUploader"] label {
    color: rgba(232,230,225,0.7) !important;
    font-size: 14px !important;
}

/* ─── RESULT BOXES ─── */
.result-row {
    display: grid; grid-template-columns: 1fr 1fr;
    gap: 14px; margin: 1.4rem 0;
}
.result-box {
    background: #111113;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 1.6rem;
    position: relative; overflow: hidden;
}
.result-box::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0; height: 2px;
}
.result-box.green::before  { background: linear-gradient(90deg,#4ADE80,#22c55e); }
.result-box.amber::before  { background: linear-gradient(90deg,#FBBF24,#F59E0B); }
.result-box.red::before    { background: linear-gradient(90deg,#F87171,#EF4444); }
.result-box.blue::before   { background: linear-gradient(90deg,#60A5FA,#3B82F6); }

.result-box-label {
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1.2px; color: rgba(232,230,225,0.4);
    margin-bottom: 0.8rem;
}
.result-pct {
    font-family: 'Sora', sans-serif;
    font-size: 3.2rem; font-weight: 700; line-height: 1;
    letter-spacing: -2px; margin-bottom: 6px;
}
.result-pct.green { color: #4ADE80; }
.result-pct.amber { color: #FBBF24; }
.result-pct.red   { color: #F87171; }
.result-badge {
    display: inline-block; font-size: 11px; font-weight: 600;
    padding: 3px 10px; border-radius: 20px; margin-top: 4px;
}
.badge-green { background: rgba(74,222,128,0.12); color: #4ADE80; }
.badge-amber { background: rgba(251,191,36,0.12); color: #FBBF24; }
.badge-red   { background: rgba(248,113,113,0.12); color: #F87171; }

.report-box {
    display: flex; flex-direction: column;
    align-items: flex-start; justify-content: space-between;
    gap: 1rem;
}
.report-box-title {
    font-size: 13px; font-weight: 600; color: #FFFFFF; margin-bottom: 4px;
}
.report-box-desc {
    font-size: 12px; color: rgba(232,230,225,0.45); line-height: 1.6;
}

/* ─── GRADCAM ─── */
.gradcam-wrap {
    background: #111113; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; overflow: hidden; margin: 1.4rem 0;
}
.gradcam-header {
    padding: 12px 18px; border-bottom: 1px solid rgba(255,255,255,0.06);
    font-size: 12px; font-weight: 600; color: rgba(232,230,225,0.5);
    text-transform: uppercase; letter-spacing: 1px;
    display: flex; justify-content: space-between; align-items: center;
}
.gradcam-legend {
    font-size: 11px; font-weight: 400; letter-spacing: 0;
    text-transform: none; color: rgba(232,230,225,0.35);
}

/* ─── SUGGESTION SECTION ─── */
.suggestion-wrap {
    background: #111113; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 1.6rem; margin-top: 1.4rem;
}
.suggestion-header {
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1.2px; color: rgba(232,230,225,0.4); margin-bottom: 1.2rem;
}
.sug-item {
    display: flex; gap: 14px; align-items: flex-start;
    padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
}
.sug-item:last-child { border-bottom: none; padding-bottom: 0; }
.sug-num {
    width: 26px; height: 26px; border-radius: 50%; flex-shrink: 0;
    display: flex; align-items: center; justify-content: center;
    font-size: 11px; font-weight: 600; margin-top: 1px;
}
.sug-num.green { background: rgba(74,222,128,0.12); color: #4ADE80; }
.sug-num.amber { background: rgba(251,191,36,0.12); color: #FBBF24; }
.sug-num.red   { background: rgba(248,113,113,0.12); color: #F87171; }
.sug-title { font-size: 13px; font-weight: 600; color: #E8E6E1; margin-bottom: 3px; }
.sug-body  { font-size: 12px; color: rgba(232,230,225,0.5); line-height: 1.65; }

/* ─── PROFILE ─── */
.profile-hero {
    background: #111113; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 2rem; margin-bottom: 1.4rem;
    display: flex; align-items: center; gap: 1.5rem;
}
.profile-avatar {
    width: 70px; height: 70px; border-radius: 50%; flex-shrink: 0;
    background: rgba(74,222,128,0.12);
    border: 2px solid rgba(74,222,128,0.3);
    display: flex; align-items: center; justify-content: center;
    font-family: 'Sora', sans-serif; font-size: 24px; font-weight: 700; color: #4ADE80;
}
.profile-name { font-family: 'Sora', sans-serif; font-size: 1.3rem; font-weight: 700; color: #FFFFFF; }
.profile-sub  { font-size: 13px; color: rgba(232,230,225,0.45); margin-top: 3px; }

.section-box {
    background: #111113; border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px; padding: 1.6rem; margin-bottom: 1.2rem;
}
.section-label {
    font-size: 10px; font-weight: 600; text-transform: uppercase;
    letter-spacing: 1.2px; color: rgba(232,230,225,0.35); margin-bottom: 1.2rem;
}

.history-row {
    display: flex; align-items: center; justify-content: space-between;
    padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.05);
}
.history-row:last-child { border-bottom: none; }
.history-date { font-size: 12px; color: rgba(232,230,225,0.45); }
.history-label { font-size: 13px; font-weight: 600; color: #E8E6E1; }
.history-pct { font-size: 13px; font-weight: 600; font-family: 'Sora', sans-serif; }

/* ─── INPUTS ─── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stTextArea > div > div > textarea,
.stSelectbox > div > div {
    background: #18181B !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 8px !important;
    color: #E8E6E1 !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}
.stTextInput label, .stTextArea label,
.stNumberInput label, .stSelectbox label {
    color: rgba(232,230,225,0.6) !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* ─── SPINNER ─── */
[data-testid="stSpinner"] p { color: rgba(232,230,225,0.6) !important; }

/* ─── st.image ─── */
[data-testid="stImage"] img {
    border-radius: 10px;
    width: 100%;
}

/* ─── DISCLAIMER ─── */
.disclaimer {
    background: rgba(251,191,36,0.06);
    border: 1px solid rgba(251,191,36,0.2);
    border-radius: 8px; padding: 10px 14px;
    font-size: 12px; color: rgba(251,191,36,0.8);
    margin-bottom: 1.4rem;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────
def get_level(dys_prob: float):
    pct = dys_prob * 100
    if pct < 40:
        return "Low Risk", "green", "green", "badge-green"
    elif pct < 70:
        return "Moderate Risk", "amber", "amber", "badge-amber"
    else:
        return "High Risk", "red", "red", "badge-red"


def get_suggestions(dys_prob: float):
    pct = dys_prob * 100
    if pct < 40:
        return [
            {"title": "Maintain reading habits",
             "body": "Daily 15-minute reading exercises help consolidate phonological awareness and fluency. Continue current practice."},
            {"title": "Cursive writing practice",
             "body": "Introduce connected cursive to further develop motor-linguistic coordination and spatial letter positioning."},
            {"title": "Routine monitoring",
             "body": "Re-assess every 3–6 months to track any changes. Early detection remains the most effective strategy."},
        ]
    elif pct < 70:
        return [
            {"title": "Letter reversal correction (b/d/p/q)",
             "body": "Use multisensory tracing cards for commonly reversed letters — 10 min/day. Combine visual, tactile, and auditory feedback simultaneously."},
            {"title": "Consistent word spacing",
             "body": "Use the finger-spacing technique between words. Practice on dotted lined paper with an emphasis on equal inter-word gaps."},
            {"title": "Phonological segmentation games",
             "body": "Break 2-syllable words into phonemes aloud while writing each letter. Clapping syllable games reinforce sound-letter mapping."},
            {"title": "Raised-line paper for baseline stability",
             "body": "Tactile baseline feedback reduces letter floating and baseline drift. Use for 20-minute daily sessions."},
        ]
    else:
        return [
            {"title": "Orton-Gillingham structured literacy",
             "body": "Systematic, explicit phonics instruction — 30 min structured sessions, 5×/week. Focus on phoneme-grapheme correspondence."},
            {"title": "Multisensory letter formation",
             "body": "Trace letters in sand or textured surfaces while pronouncing each phoneme aloud. Sky-writing reinforces motor memory. Prioritise most-confused pairs first."},
            {"title": "Chunking & deliberate word spacing",
             "body": "Write one word at a time with a deliberate pause. Use colour-coded word separators. Practice with block-letter stamps before transitioning to cursive."},
            {"title": "Slow dictation & phoneme mapping",
             "body": "Hear word → segment phonemes aloud → write letter by letter. Record and replay sessions for self-monitoring and progress review."},
            {"title": "Specialist referral",
             "body": "A score at this level warrants formal assessment by an educational psychologist. Combine AI screening with clinical evaluation for best outcomes."},
        ]


# ─────────────────────────────────────────────────
# NAV
# ─────────────────────────────────────────────────
def render_nav():
    p = st.session_state.page
    nav_html = f"""
    <div class="nav">
      <div class="nav-logo">Dysl<span>eX</span>pert</div>
      <div class="nav-links">
        <span class="nav-link {'active' if p=='home' else ''}">Home</span>
        <span class="nav-link {'active' if p=='analyze' else ''}">Analyse</span>
        <span class="nav-link {'active' if p=='profile' else ''}">Profile</span>
      </div>
    </div>
    """
    st.markdown(nav_html, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns([5, 1, 1, 1])
    with c2:
        if st.button("Home", key="n_home"):
            st.session_state.page = "home"; st.rerun()
    with c3:
        if st.button("Analyse", key="n_analyze"):
            st.session_state.page = "analyze"; st.rerun()
    with c4:
        if st.button("Profile", key="n_profile"):
            st.session_state.page = "profile"; st.rerun()


# ─────────────────────────────────────────────────
# PAGE 1 — HOME
# ─────────────────────────────────────────────────
def page_home():
    st.markdown("""
    <div class="home-wrap">
      <div class="home-chip">AI · Handwriting Analysis · Grad-CAM XAI</div>
      <div class="home-h1">Early dyslexia detection<br>powered by <em>deep learning</em></div>
      <div class="home-sub">
        Upload a handwriting sample and receive an instant AI-powered screening report —
        complete with Grad-CAM visualisation, risk level, and personalised intervention suggestions.
      </div>
    </div>
    """, unsafe_allow_html=True)

    # CTA row
    c1, c2, c3, c4 = st.columns([2, 1, 1, 2])
    with c2:
        if st.button("Start Analysis →", key="home_analyze"):
            st.session_state.page = "analyze"; st.rerun()
    with c3:
        st.markdown('<div class="ghost-btn">', unsafe_allow_html=True)
        if st.button("My Profile", key="home_profile"):
            st.session_state.page = "profile"; st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="home-cards" style="margin-top:3rem;max-width:820px;margin-left:auto;margin-right:auto;">
      <div class="home-card">
        <div class="home-card-icon">🧠</div>
        <h3>ResNet18 + Grad-CAM</h3>
        <p>Deep neural network trained on real handwriting datasets, with gradient-weighted activation maps for full transparency.</p>
      </div>
      <div class="home-card">
        <div class="home-card-icon">📊</div>
        <h3>Instant risk level</h3>
        <p>Get a clear Low / Moderate / High dyslexia risk score with exact probability percentage — no ambiguity.</p>
      </div>
      <div class="home-card">
        <div class="home-card-icon">✏️</div>
        <h3>Personalised suggestions</h3>
        <p>Targeted handwriting and phonological exercises generated automatically based on the detected risk level.</p>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────
# PAGE 2 — ANALYSE
# ─────────────────────────────────────────────────
def page_analyze():
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Handwriting Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Upload a handwriting image to receive an AI-powered dyslexia screening.</div>', unsafe_allow_html=True)
    st.markdown('<div class="disclaimer">⚠️  Research prototype — results must not replace a formal clinical diagnosis.</div>', unsafe_allow_html=True)

    # ── Upload box ──
    uploaded = st.file_uploader(
        "Drop a handwriting image here  ·  JPG / PNG",
        type=["jpg", "jpeg", "png"],
        label_visibility="visible",
        key="uploader",
    )

    if uploaded is not None:
        temp_path = "temp.jpg"
        with open(temp_path, "wb") as f:
            f.write(uploaded.getbuffer())

        # Run model
        with st.spinner("Analysing handwriting with ResNet18..."):
            label, dys_prob, non_dys_prob = predict(temp_path)
            dys_prob = float(dys_prob)
            non_dys_prob = float(non_dys_prob)

        with st.spinner("Generating Grad-CAM heatmap..."):
            try:
                gradcam_path, _, _ = run_gradcam(temp_path)
                has_gradcam = True
            except Exception:
                has_gradcam = False
                gradcam_path = None

        level_text, level_color, pct_color, badge_cls = get_level(dys_prob)
        dys_pct  = round(dys_prob * 100, 1)
        non_pct  = round(non_dys_prob * 100, 1)

        # ── Grad-CAM images row ──
        img_c1, img_c2 = st.columns(2)
        with img_c1:
            st.markdown("""
            <div class="gradcam-wrap">
              <div class="gradcam-header">Original sample</div>
            </div>
            """, unsafe_allow_html=True)
            st.image(uploaded, use_container_width=True)

        with img_c2:
            st.markdown(f"""
            <div class="gradcam-wrap">
              <div class="gradcam-header">
                Grad-CAM heatmap
                <span class="gradcam-legend">Red = high relevance · Blue = low</span>
              </div>
            </div>
            """, unsafe_allow_html=True)
            if has_gradcam:
                st.image(gradcam_path, use_container_width=True)
            else:
                st.markdown('<p style="color:rgba(248,113,113,0.7);font-size:13px;padding:1rem;">Grad-CAM unavailable — check layer name in gradcam_visualize.py</p>', unsafe_allow_html=True)

        # ── Result boxes row: Dyslexia Level | Report ──
        st.markdown(f"""
        <div class="result-row">

          <div class="result-box {level_color}">
            <div class="result-box-label">Dyslexia risk level</div>
            <div class="result-pct {pct_color}">{dys_pct}%</div>
            <div style="font-size:12px;color:rgba(232,230,225,0.45);margin:6px 0 10px;">
              Dyslexic probability
            </div>
            <span class="result-badge {badge_cls}">{level_text}</span>

            <div style="margin-top:1.2rem;">
              <div style="display:flex;justify-content:space-between;font-size:11px;
                          color:rgba(232,230,225,0.4);margin-bottom:5px;">
                <span>Non-Dyslexic</span><span>{non_pct}%</span>
              </div>
              <div style="height:5px;background:rgba(255,255,255,0.07);border-radius:3px;overflow:hidden;">
                <div style="height:100%;width:{non_pct}%;background:#4ADE80;border-radius:3px;"></div>
              </div>
              <div style="display:flex;justify-content:space-between;font-size:11px;
                          color:rgba(232,230,225,0.4);margin:8px 0 5px;">
                <span>Dyslexic</span><span>{dys_pct}%</span>
              </div>
              <div style="height:5px;background:rgba(255,255,255,0.07);border-radius:3px;overflow:hidden;">
                <div style="height:100%;width:{dys_pct}%;
                  background:{'#F87171' if level_color=='red' else '#FBBF24' if level_color=='amber' else '#4ADE80'};
                  border-radius:3px;"></div>
              </div>
            </div>
          </div>

          <div class="result-box blue">
            <div class="result-box-label">Analysis report</div>
            <div class="report-box-title">PDF Report</div>
            <div class="report-box-desc">
              Includes prediction result, probability breakdown, Grad-CAM heatmap image,
              and clinical interpretation — ready to share with a therapist or educator.
            </div>
          </div>

        </div>
        """, unsafe_allow_html=True)

        # Download button rendered inside the blue box area
        col_sp1, col_dl, col_sp2 = st.columns([1.05, 0.95, 1])
        with col_dl:
            if has_gradcam:
                rpt = generate_report(label, dys_prob, non_dys_prob, gradcam_path)
                with open(rpt, "rb") as f:
                    st.download_button(
                        "⬇  Download PDF Report",
                        data=f,
                        file_name=f"dyslexia_report_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.pdf",
                        mime="application/pdf",
                        key="dl_pdf",
                    )
            else:
                st.markdown('<p style="font-size:12px;color:rgba(248,113,113,0.6);">Report unavailable without Grad-CAM.</p>', unsafe_allow_html=True)

        # ── Save to profile ──
        col_sv, _ = st.columns([1, 3])
        with col_sv:
            if st.button("💾  Save to Profile", key="save_btn"):
                st.session_state.sessions.append({
                    "date": datetime.datetime.now().strftime("%d %b %Y  %H:%M"),
                    "label": label,
                    "dys_pct": dys_pct,
                    "non_pct": non_pct,
                    "level": level_text,
                    "level_color": level_color,
                })
                st.success("Saved to your profile.")

        # ── Suggestions ──
        suggestions = get_suggestions(dys_prob)
        num_color = level_color

        sug_html = """
        <div class="suggestion-wrap">
          <div class="suggestion-header">Analysis-based suggestions</div>
        """
        for i, s in enumerate(suggestions):
            sug_html += f"""
            <div class="sug-item">
              <div class="sug-num {num_color}">{i+1}</div>
              <div>
                <div class="sug-title">{s['title']}</div>
                <div class="sug-body">{s['body']}</div>
              </div>
            </div>
            """
        sug_html += "</div>"
        st.markdown(sug_html, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────
# PAGE 3 — PROFILE
# ─────────────────────────────────────────────────
def page_profile():
    st.markdown('<div class="page-wrap">', unsafe_allow_html=True)
    st.markdown('<div class="page-title">Profile</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub">Your personal details and test history.</div>', unsafe_allow_html=True)

    prof = st.session_state.profile
    sessions = st.session_state.sessions

    # ── Avatar hero ──
    initials = (prof["name"][:2].upper() if len(prof["name"]) >= 2
                else prof["name"][:1].upper() if prof["name"] else "?")
    grade_str = f"Grade {prof['grade']}  ·  {prof['school']}" if prof["grade"] or prof["school"] else "No details yet"

    st.markdown(f"""
    <div class="profile-hero">
      <div class="profile-avatar">{initials}</div>
      <div>
        <div class="profile-name">{prof['name'] if prof['name'] else 'Unnamed User'}</div>
        <div class="profile-sub">{grade_str}</div>
        <div class="profile-sub" style="margin-top:2px;">
          {f"Age: {prof['age']}" if prof['age'] else ''}
          {f"  ·  Therapist: {prof['therapist']}" if prof['therapist'] else ''}
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Edit details ──
    with st.expander("✏️  Edit personal details", expanded=(not prof["name"])):
        c1, c2 = st.columns(2)
        with c1:
            n  = st.text_input("Full name",       value=prof["name"],       key="p_name")
            a  = st.text_input("Age",              value=prof["age"],        key="p_age")
            g  = st.text_input("Grade / Class",    value=prof["grade"],      key="p_grade")
        with c2:
            sc = st.text_input("School / Institute", value=prof["school"],   key="p_school")
            th = st.text_input("Therapist / Teacher", value=prof["therapist"], key="p_therapist")
        nt = st.text_area("Notes",                 value=prof["notes"],      key="p_notes", height=80)

        if st.button("Save details", key="save_profile"):
            st.session_state.profile.update({
                "name": n, "age": a, "grade": g,
                "school": sc, "therapist": th, "notes": nt,
            })
            st.success("Profile saved!")
            st.rerun()

    # ── Test history ──
    st.markdown("""
    <div class="section-box" style="margin-top:1.2rem;">
      <div class="section-label">Test history</div>
    """, unsafe_allow_html=True)

    if not sessions:
        st.markdown("""
        <p style="font-size:13px;color:rgba(232,230,225,0.35);padding:1rem 0;">
          No tests taken yet. Head to <strong style="color:#E8E6E1;">Analyse</strong> to run your first screening.
        </p>
        """, unsafe_allow_html=True)
        if st.button("Go to Analyse →", key="prof_goto"):
            st.session_state.page = "analyze"; st.rerun()
    else:
        color_map = {"green": "#4ADE80", "amber": "#FBBF24", "red": "#F87171"}
        for s in reversed(sessions):
            col = color_map.get(s.get("level_color", "green"), "#4ADE80")
            st.markdown(f"""
            <div class="history-row">
              <div>
                <div class="history-label">{s['label']}</div>
                <div class="history-date">{s['date']}</div>
              </div>
              <div style="text-align:right;">
                <div class="history-pct" style="color:{col};">{s['dys_pct']}%</div>
                <div style="font-size:11px;color:rgba(232,230,225,0.35);">{s['level']}</div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Trend chart
        if len(sessions) >= 2:
            import pandas as pd
            st.markdown('</div>', unsafe_allow_html=True)  # close section-box first
            st.markdown("""
            <div class="section-box" style="margin-top:1.2rem;">
              <div class="section-label">Risk trend over time</div>
            """, unsafe_allow_html=True)
            df = pd.DataFrame({
                "Test": [f"#{i+1}" for i in range(len(sessions))],
                "Dyslexic %": [s["dys_pct"] for s in sessions],
            }).set_index("Test")
            st.line_chart(df, color="#4ADE80", height=180)

        col_clr, _ = st.columns([1, 3])
        with col_clr:
            if st.button("Clear history", key="clr_hist"):
                st.session_state.sessions = []
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # ── Notes section ──
    if prof["notes"]:
        st.markdown(f"""
        <div class="section-box" style="margin-top:1.2rem;">
          <div class="section-label">Notes</div>
          <p style="font-size:13px;color:rgba(232,230,225,0.6);line-height:1.7;">{prof['notes']}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────
# ROUTER
# ─────────────────────────────────────────────────
render_nav()

pg = st.session_state.page
if   pg == "home":    page_home()
elif pg == "analyze": page_analyze()
elif pg == "profile": page_profile()'''
