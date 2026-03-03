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