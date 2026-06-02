import streamlit as st
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
from PIL import Image
import tempfile
import cv2
import os

# Page Config
st.set_page_config(
    page_title="Helmet Detection System",
    page_icon="🪖",
    layout="wide"
)

st.title("🪖 Helmet Detection System")
st.write("Upload an image or video to detect helmets using YOLOv8.")

# Load Model
@st.cache_resource
def load_model():
    model_path = hf_hub_download(
        repo_id="iam-tsr/yolov8n-helmet-detection",
        filename="best.pt"
    )
    return YOLO(model_path)

model = load_model()

# Sidebar
st.sidebar.header("Settings")
confidence = st.sidebar.slider(
    "Confidence Threshold",
    0.1,
    1.0,
    0.5,
    0.1
)

option = st.radio(
    "Select Input Type",
    ["Image", "Video"]
)

# ---------------- IMAGE DETECTION ---------------- #

if option == "Image":

    uploaded_file = st.file_uploader(
        "Upload an Image",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)

        with st.spinner("Detecting helmets..."):

            results = model.predict(
                image,
                conf=confidence
            )

            detected_img = results[0].plot()

        with col2:
            st.subheader("Detection Result")
            st.image(
                detected_img,
                use_container_width=True
            )

        st.success("Detection completed!")

# ---------------- VIDEO DETECTION ---------------- #

elif option == "Video":

    uploaded_video = st.file_uploader(
        "Upload a Video",
        type=["mp4", "avi", "mov"]
    )

    if uploaded_video is not None:

        st.video(uploaded_video)

        temp_video = tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".mp4"
        )

        temp_video.write(uploaded_video.read())
        temp_video.close()

        st.info("Processing video. This may take a few moments...")

        with st.spinner("Running helmet detection..."):

            results = model.predict(
                source=temp_video.name,
                conf=confidence,
                save=True
            )

        st.success("Video processed successfully!")

        # Locate latest output video
        output_dir = "runs/detect"

        if os.path.exists(output_dir):

            folders = sorted(
                [os.path.join(output_dir, f)
                 for f in os.listdir(output_dir)],
                key=os.path.getmtime
            )

            latest_folder = folders[-1]

            video_files = [
                f for f in os.listdir(latest_folder)
                if f.endswith((".mp4", ".avi", ".mov"))
            ]

            if video_files:

                output_video_path = os.path.join(
                    latest_folder,
                    video_files[0]
                )

                st.subheader("Processed Video")

                st.video(output_video_path)

                with open(output_video_path, "rb") as file:
                    st.download_button(
                        "⬇ Download Processed Video",
                        file,
                        file_name="helmet_detection_output.mp4"
                    )

# Footer
st.markdown("---")
st.markdown(
    "Developed using **YOLOv8 + Streamlit + Deep Learning**"
)
