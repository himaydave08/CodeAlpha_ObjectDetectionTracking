import streamlit as st
import os
import base64

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
    try:
        # Load and convert the video file to base64 for embedding in raw HTML
        with open(video_path, "rb") as f:
            video_bytes = f.read()
        video_base64 = base64.b64encode(video_bytes).decode("utf-8")
        
        # HTML5 Video Tag: autoplay, loop, muted, playsinline, and NO controls attribute
        video_html = f"""
        <div style="display: flex; justify-content: center; width: 100%;">
            <video autoplay loop muted playsinline style="width: 100%; max-width: 800px; border-radius: 12px; box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);">
                <source src="data:video/mp4;base64,{video_base64}" type="video/mp4">
                Your browser does not support the video tag.
            </video>
        </div>
        """
        st.markdown(video_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Error loading video player: {e}")
else:
    st.warning("Output video not found under output/output_video.mp4")
