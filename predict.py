
from pathlib import Path
import csv
import time

import torch
from ultralytics import YOLO


model_name = "yolo11m"

PROJECT_DIR = Path.home() / "Projetos" / "YoloProject"

MODEL_PATH = (
    PROJECT_DIR / "runs" / "detect" / model_name
    / "weights" / "best.pt"
)

IMAGES_DIR = PROJECT_DIR / "dataset" / "images" / "test"
LABELS_DIR = PROJECT_DIR / "dataset" / "labels" / "test"

file = model_name + "_resultado_teste.csv"
CSV_PATH = PROJECT_DIR / file

CONFIDENCE = 0.25
DEVICE = 0 if torch.cuda.is_available() else "cpu"

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}

def contar_labels(label_path, num_classes):
    """Conta as instâncias de cada classe no arquivo YOLO."""
    counts = [0] * num_classes

    if not label_path.exists():
        return counts, False

    with label_path.open("r", encoding="utf-8") as file:
        for line in file:
            parts = line.strip().split()

            if not parts:
                continue

            try:
                class_id = int(parts[0])
            except ValueError:
                continue

            if 0 <= class_id < num_classes:
                counts[class_id] += 1

    return counts, True


def contar_deteccoes(result, num_classes):
    """Conta as instâncias detectadas pelo modelo por classe."""
    counts = [0] * num_classes

    if result.boxes is None or len(result.boxes) == 0:
        return counts

    class_ids = result.boxes.cls.int().cpu().tolist()

    for class_id in class_ids:
        if 0 <= class_id < num_classes:
            counts[class_id] += 1

    return counts


def sincronizar_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def main():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Modelo não encontrado: {MODEL_PATH}")

    if not IMAGES_DIR.is_dir():
        raise FileNotFoundError(f"Imagens não encontradas: {IMAGES_DIR}")

    if not LABELS_DIR.is_dir():
        raise FileNotFoundError(f"Labels não encontrados: {LABELS_DIR}")

    images = sorted(
        path for path in IMAGES_DIR.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise RuntimeError("Nenhuma imagem encontrada no conjunto de teste.")

    model = YOLO(str(MODEL_PATH))

    # Usa os nomes e IDs definidos no modelo treinado.
    names = model.names
    if isinstance(names, dict):
        class_names = [names[i] for i in sorted(names)]
    else:
        class_names = list(names)

    num_classes = len(class_names)

    print(f"Modelo: {MODEL_PATH}")
    print(f"Dispositivo: {DEVICE}")
    print(f"Imagens para teste: {len(images)}")
    print(f"Classes do modelo: {num_classes}")

    if num_classes != 4:
        print(
            f"ATENÇÃO: o modelo possui {num_classes} classes, "
            "e não 4. O CSV usará as classes existentes no modelo."
        )

    # Aquecimento: evita incluir parte do custo inicial
    # de carregamento/inicialização na primeira medição.
    model.predict(
        source=str(images[0]),
        conf=CONFIDENCE,
        device=DEVICE,
        verbose=False,
        save=False,
    )
    sincronizar_gpu()

    # Cabeçalho: nome, tempo total, pares por classe, inferência.
    columns = ["arquivo", "tempo_total_ms"]

    for class_id, class_name in enumerate(class_names):
        safe_name = str(class_name).strip().replace(" ", "_")
        columns.append(f"{class_id}_{safe_name}_detectado")
        columns.append(f"{class_id}_{safe_name}_label")

    columns.append("tempo_inferencia_ms")

    CSV_PATH.parent.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open(
        "w", newline="", encoding="utf-8-sig"
    ) as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()

        for index, image_path in enumerate(images, start=1):
            label_path = LABELS_DIR / f"{image_path.stem}.txt"

            # Contagem real das instâncias anotadas.
            label_counts, label_exists = contar_labels(
                label_path, num_classes
            )

            # Mede o tempo total de processamento por imagem.
            sincronizar_gpu()
            start = time.perf_counter()

            results = model.predict(
                source=str(image_path),
                conf=CONFIDENCE,
                device=DEVICE,
                verbose=False,
                save=False,
            )

            sincronizar_gpu()
            elapsed_ms = (time.perf_counter() - start) * 1000

            result = results[0]

            # Tempo de inferência informado pelo Ultralytics.
            inference_ms = result.speed.get("inference", 0.0)

            detected_counts = contar_deteccoes(
                result, num_classes
            )

            row = {
                "arquivo": image_path.name,
                "tempo_total_ms": round(elapsed_ms, 3),
                "tempo_inferencia_ms": round(inference_ms, 3),
            }

            for class_id, class_name in enumerate(class_names):
                safe_name = str(class_name).strip().replace(" ", "_")

                row[f"{class_id}_{safe_name}_detectado"] = (
                    detected_counts[class_id]
                )
                row[f"{class_id}_{safe_name}_label"] = (
                    label_counts[class_id]
                )

            writer.writerow(row)

            label_status = "" if label_exists else " [SEM LABEL]"

            print(
                f"[{index}/{len(images)}] {image_path.name} | "
                f"Total: {elapsed_ms:.2f} ms | "
                f"Inferência: {inference_ms:.2f} ms"
                f"{label_status}"
            )

    print(f"\nTeste concluído. CSV salvo em:\n{CSV_PATH}")


if __name__ == "__main__":
    main()