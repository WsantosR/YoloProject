from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent


def main():
    # Modelo YOLOv8 Nano pré-treinado
    model = YOLO("yolov8n.pt")

    model.train(
    data=str(BASE_DIR / "data.yaml"),

    epochs=200,
    imgsz=640,
    batch=16,

    name="cabo_p10",

    workers=8,
    device=0,

    save=True,
    val=True,

    degrees=15,
    translate=0.1,
    scale=0.5,
    fliplr=0.5,
    flipud=0.0,

    patience=30,
)


if __name__ == "__main__":
    main()