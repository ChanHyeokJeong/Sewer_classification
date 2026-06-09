import os
import shutil
import random
import argparse
from pathlib import Path

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

DAMAGED_CLASSES = [
    "Broken_Pipe",
    "Crack",
    "Joint_Faulty",
    "Joint_Displaced",
    "Lateral_Protruding",
    "Surface_Damage",
    "Deposits_Silty",
    "Etc"
]

NON_DAMAGED_CLASSES = [
    "Inside",
    "Outside",
    "Pipe_Joint"
]


def collect_images(input_dir, class_list):
    input_dir = Path(input_dir)
    files = []

    for class_name in class_list:
        class_dir = input_dir / class_name

        if not class_dir.exists():
            print(f"Skip missing folder: {class_dir}")
            continue

        for path in class_dir.rglob("*"):
            if path.suffix.lower() in IMAGE_EXTS:
                files.append(path)

    return files


def copy_files(files, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, src in enumerate(files):
        dst = output_dir / f"{idx:07d}_{src.name}"
        shutil.copy2(src, dst)


def split_files(files, train_n, val_n, test_n, seed):
    random.seed(seed)
    files = files.copy()
    random.shuffle(files)

    required_n = train_n + val_n + test_n

    if len(files) < required_n:
        raise ValueError(f"Required {required_n} images, but found {len(files)} images.")

    train_files = files[:train_n]
    val_files = files[train_n:train_n + val_n]
    test_files = files[train_n + val_n:required_n]

    return train_files, val_files, test_files


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, default="sewer_full_classified")
    parser.add_argument("--out_dir", type=str, default="data_sewer_binary_large")
    parser.add_argument("--train_per_class", type=int, default=1000)
    parser.add_argument("--val_per_class", type=int, default=200)
    parser.add_argument("--test_per_class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    groups = {
        "Damaged": DAMAGED_CLASSES,
        "Non_Damaged": NON_DAMAGED_CLASSES
    }

    for binary_label, class_list in groups.items():
        files = collect_images(args.input_dir, class_list)

        print(f"\n{binary_label}")
        print("Collected images:", len(files))

        train_files, val_files, test_files = split_files(
            files,
            args.train_per_class,
            args.val_per_class,
            args.test_per_class,
            args.seed
        )

        copy_files(train_files, Path(args.out_dir) / "train" / binary_label)
        copy_files(val_files, Path(args.out_dir) / "val" / binary_label)
        copy_files(test_files, Path(args.out_dir) / "test" / binary_label)

        print("Train:", len(train_files))
        print("Val:", len(val_files))
        print("Test:", len(test_files))

    print("\nBinary split completed.")
    print("Output folder:", args.out_dir)


if __name__ == "__main__":
    main()