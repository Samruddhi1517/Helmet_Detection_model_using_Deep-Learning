import streamlit as st

# MUST BE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Helmet Detection Dashboard",
    page_icon="🪖",
    layout="wide"
)

from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image
import tempfile
import os

# ===== CUSTOM CSS =====
st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
}

h1, h2, h3, p, label {
    color: white !important;
}

[data-testid="stFileUploader"] {
    border: 2px dashed #38bdf8;
    border-radius: 10px;
    padding: 20px;
    background: #1e293b;
}

.stButton button {
    background: #38bdf8 !important;
    color: white !important;
    border-radius: 8px;
    border: none;
}

[data-testid="stSidebar"] {
    background-color: #1e293b;
}
</style>
""", unsafe_allow_html=True)

st.title("🚦 Helmet Detection Dashboard")
st.write("Upload an image or video for helmet detection.")
