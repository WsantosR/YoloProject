from ultralytics import YOLO

models = [
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
]

for model_name in models:
    print(f"Downloading {model_name}...")
    YOLO(model_name)
    print(f"{model_name} OK")