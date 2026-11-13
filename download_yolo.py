from ultralytics import YOLO

models = [
    # YOLOv8
    "yolov8n.pt",
    "yolov8s.pt",
    "yolov8m.pt",

    # YOLOv9
    "yolov9t.pt",
    "yolov9s.pt",
    "yolov9m.pt",

    # YOLOv10
    "yolov10n.pt",
    "yolov10s.pt",
    "yolov10m.pt",

    # YOLO11
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
]

for model_name in models:
    print(f"Downloading {model_name}...")
    YOLO(model_name)
    print(f"{model_name} OK")