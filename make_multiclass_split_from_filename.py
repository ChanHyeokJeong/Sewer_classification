import os
import shutil
import random
import argparse
from pathlib import Path
from collections import Counter

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

PREFIX_TO_CLASS = {
    "BK": "Broken_Pipe",
    "CC": "Crack",
    "CL": "Crack",
    "LP": "Lateral_Protruding",
    "SD": "Surface_Damage",
    "IN": "Inside",
    "OUT": "Outside",
    "JF": "Joint_Faulty",
    "PJ": "Pipe_Joint"
}


def get_prefix(filename):
    if "_" not in filename:
        return None
    return filename.split("_")[0].upper()


def collect_images_by_class(input_dir):
    input_dir = Path(input_dir)

    class_to_files = {}
    prefix_counter = Counter()
    skipped_files = []

    for path in input_dir.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        prefix = get_prefix(path.name)

        if prefix is None or prefix not in PREFIX_TO_CLASS:
            skipped_files.append(path)
            continue

        class_name = PREFIX_TO_CLASS[prefix]

        if class_name not in class_to_files:
            class_to_files[class_name] = []

        class_to_files[class_name].append(path)
        prefix_counter[prefix] += 1

    return class_to_files, prefix_counter, skipped_files


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


def copy_files(files, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for idx, src in enumerate(files):
        dst = output_dir / f"{idx:07d}_{src.name}"
        shutil.copy2(src, dst)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_dir", type=str, default="sewer_full_raw")
    parser.add_argument("--out_dir", type=str, default="data_sewer_multiclass_large")
    parser.add_argument("--train_per_class", type=int, default=1000)
    parser.add_argument("--val_per_class", type=int, default=200)
    parser.add_argument("--test_per_class", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    class_to_files, prefix_counter, skipped_files = collect_images_by_class(args.input_dir)

    print("\nPrefix counts")
    for key, value in sorted(prefix_counter.items()):
        print(f"{key}: {value}")

    print("\nClass counts")
    for class_name, files in sorted(class_to_files.items()):
        print(f"{class_name}: {len(files)}")

    print("\nSkipped files:", len(skipped_files))

    out_dir = Path(args.out_dir)

    if out_dir.exists():
        print(f"\nWarning: output folder already exists: {out_dir}")
        print("Delete it manually if you want a clean split.")

    for class_name, files in sorted(class_to_files.items()):
        print(f"\nProcessing {class_name}")

        try:
            train_files, val_files, test_files = split_files(
                files,
                args.train_per_class,
                args.val_per_class,
                args.test_per_class,
                args.seed
            )
        except ValueError as e:
            print(f"Skip {class_name}: {e}")
            continue

        copy_files(train_files, out_dir / "train" / class_name)
        copy_files(val_files, out_dir / "val" / class_name)
        copy_files(test_files, out_dir / "test" / class_name)

        print("Train:", len(train_files))
        print("Val:", len(val_files))
        print("Test:", len(test_files))

    print("\nMulti-class split completed.")
    print("Output folder:", out_dir)


if __name__ == "__main__":
    main()