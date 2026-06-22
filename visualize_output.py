import streamlit as st
import os

st.set_page_config(
    page_title="Object Detection & Tracking",
    page_icon="🔍",
    layout="centered"
)

# Custom minimal styling
st.markdown("""
<style>
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-top: 20px;
        margin-bottom: 30px;
        font-size: 2.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown("<h1 class='main-title'>Object Detection Tracking</h1>", unsafe_allow_html=True)

# Output Video Player
video_path = os.path.join("output", "output_video.mp4")

if os.path.exists(video_path):
    st.video(video_path, format="video/mp4")
else:
    st.warning("Output video not found under output/output_video.mp4")
