from ultralytics import YOLO


model = YOLO(
    "/home/willian/Projetos/codigos_python/runs/detect/cabo_p10-8/weights/best.pt"
)

results = model.predict(
    source="/home/willian/Projetos/codigos_python/test_yolo/dataset/images/train/618beb24-IMG_1184.JPG",
    imgsz=640,
    conf=0.5,

    # Pasta onde o resultado será salvo
    project="/home/willian/Projetos/YoloProject/tmp",

    save=True,
)

for result in results:
    for box in result.boxes:
        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        print(
            f"Classe: {class_id} "
            f"Confiança: {confidence:.2f}"
        )