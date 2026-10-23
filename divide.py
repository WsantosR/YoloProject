
from pathlib import Path
import random
import shutil

# Pasta principal do dataset
DATASET_DIR = Path("dataset")

IMAGES_DIR = DATASET_DIR / "images"
LABELS_DIR = DATASET_DIR / "labels"

# Proporções de divisão
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Semente fixa para reproduzir a mesma divisão
random.seed(42)

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"
}


def main():
    if not IMAGES_DIR.is_dir() or not LABELS_DIR.is_dir():
        raise FileNotFoundError(
            "Não encontrei as pastas dataset/images e dataset/labels."
        )

    # Considera apenas arquivos diretamente nas pastas de origem
    images = sorted(
        file for file in IMAGES_DIR.iterdir()
        if file.is_file() and file.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise RuntimeError("Nenhuma imagem encontrada em dataset/images.")

    random.shuffle(images)

    total = len(images)
    train_end = int(total * TRAIN_RATIO)
    val_end = train_end + int(total * VAL_RATIO)

    splits = {
        "train": images[:train_end],
        "val": images[train_end:val_end],
        "test": images[val_end:],
    }

    # Criar as pastas de destino
    for split in splits:
        (IMAGES_DIR / split).mkdir(parents=True, exist_ok=True)
        (LABELS_DIR / split).mkdir(parents=True, exist_ok=True)

    missing_labels = []
    moved_images = 0
    moved_labels = 0

    for split, split_images in splits.items():
        for image_path in split_images:
            label_path = LABELS_DIR / f"{image_path.stem}.txt"

            # Verifica se a imagem e o label existem antes de mover
            if not image_path.exists():
                continue

            if label_path.exists():
                shutil.move(
                    str(label_path),
                    str(LABELS_DIR / split / label_path.name)
                )
                moved_labels += 1
            else:
                missing_labels.append(image_path.name)

            shutil.move(
                str(image_path),
                str(IMAGES_DIR / split / image_path.name)
            )
            moved_images += 1

    print("\nDivisão concluída!")
    print(f"Total de imagens: {total}")

    for split, split_images in splits.items():
        print(f"{split}: {len(split_images)} imagens")

    print(f"\nImagens movidas: {moved_images}")
    print(f"Labels movidos: {moved_labels}")

    if missing_labels:
        print(f"\nAviso: {len(missing_labels)} imagens sem label correspondente:")
        for name in missing_labels:
            print(f"  - {name}")

    # Identificar labels que não possuem imagem correspondente
    image_stems = {image.stem for image in images}
    orphan_labels = [
        label.name for label in LABELS_DIR.iterdir()
        if label.is_file()
        and label.suffix.lower() == ".txt"
        and label.stem not in image_stems
    ]

    if orphan_labels:
        print(f"\nAviso: {len(orphan_labels)} labels sem imagem correspondente.")
        for name in orphan_labels:
            print(f"  - {name}")


if __name__ == "__main__":
    main()