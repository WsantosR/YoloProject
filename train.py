from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent

def main():

    model = YOLO("yolov10s.pt")

    model.train(
    data=str(BASE_DIR / "data.yaml"),

    epochs=100,
    imgsz=640,
    batch=16,

    name="yolov10s",

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