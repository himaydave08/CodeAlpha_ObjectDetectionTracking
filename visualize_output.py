import streamlit as st
import os

st.set_page_config(
    page_title="Object Detection & Tracking Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom premium CSS
st.markdown("""
<style>
    /* Dark dashboard styles */
    .reportview-container {
        background: #0f172a;
    }
    .main-title {
        font-family: 'Outfit', 'Inter', sans-serif;
        font-weight: 800;
        background: linear-gradient(135deg, #818cf8 0%, #6366f1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
        font-size: 2.8rem;
    }
    .sub-title {
        font-family: 'Inter', sans-serif;
        color: #94a3b8;
        text-align: center;
        font-size: 1.1rem;
        margin-bottom: 30px;
    }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.1), 0 2px 4px -2px rgb(0 0 0 / 0.1);
        margin-bottom: 20px;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #818cf8;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .feature-tag {
        display: inline-block;
        background: #312e81;
        color: #c7d2fe;
        border: 1px solid #4338ca;
        padding: 4px 12px;
        border-radius: 9999px;
        font-size: 0.85rem;
        margin-right: 8px;
        margin-bottom: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar configurations
with st.sidebar:
    st.markdown("<h3 style='color: #818cf8;'>⚙️ Pipeline Metadata</h3>", unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("**Model Details**")
    st.markdown("- **Object Detector**: YOLOv8n (Nano)")
    st.markdown("- **Weights**: `yolov8n.pt` (~6.2 MB)")
    st.markdown("- **Tracker**: Deep SORT")
    st.markdown("- **Embedder**: MobileNetV2")
    
    st.markdown("---")
    st.markdown("**Video Source Details**")
    st.markdown("- **Resolution**: 768 x 432")
    st.markdown("- **Frame Count**: 596 frames")
    st.markdown("- **Playback Rate**: 12.0 FPS")
    
    st.markdown("---")
    st.markdown("<small>Internship Project | CodeAlpha AI</small>", unsafe_allow_html=True)

# Main layout
st.markdown("<h1 class='main-title'>Object Detection Tracking</h1>", unsafe_allow_html=True)
st.markdown("<p class='sub-title'>Real-Time YOLOv8 + Deep SORT Multi-Object Tracker</p>", unsafe_allow_html=True)

col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("<h4 style='color: #f8fafc;'>🎥 Tracking Playback Demo</h4>", unsafe_allow_html=True)
    video_path = os.path.join("output", "output_video.mp4")
    
    if os.path.exists(video_path):
        # Specifying format="video/mp4" to ensure HTML5 player compatibility
        st.video(video_path, format="video/mp4")
        st.caption("Annotated output video showing persistent colored bounding boxes and track IDs.")
    else:
        st.warning("🔄 Output video is currently generating or not found. Please wait a few seconds and refresh.")

with col2:
    st.markdown("<h4 style='color: #f8fafc;'>📊 Key Metrics</h4>", unsafe_allow_html=True)
    
    # Custom HTML metrics
    st.markdown("""
    <div class='metric-card'>
        <div class='metric-value'>YOLOv8n</div>
        <div class='metric-label'>Detection Model</div>
    </div>
    <div class='metric-card'>
        <div class='metric-value'>Deep SORT</div>
        <div class='metric-label'>Tracking Model</div>
    </div>
    <div class='metric-card'>
        <div class='metric-value'>596</div>
        <div class='metric-label'>Total Frames Processed</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")
st.markdown("<h4 style='color: #f8fafc;'>🌟 Pipeline Features</h4>", unsafe_allow_html=True)

st.markdown("""
<span class='feature-tag'>Real-Time Inference</span>
<span class='feature-tag'>Webcam Input Support</span>
<span class='feature-tag'>Offline Video Processing</span>
<span class='feature-tag'>Persistent Track IDs</span>
<span class='feature-tag'>Dynamic Bounding Box Coloring</span>
<span class='feature-tag'>H.264 Web-Playable Output</span>
""", unsafe_allow_html=True)
