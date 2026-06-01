from ultralytics import YOLO
from huggingface_hub import hf_hub_download

def load_model():
    pt_path = hf_hub_download(
        repo_id="iam-tsr/yolov8n-helmet-detection",
        filename="best.pt"
    )
    model = YOLO(pt_path)
    return model