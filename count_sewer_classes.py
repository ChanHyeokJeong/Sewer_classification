import argparse
from pathlib import Path
from collections import Counter

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]

LABEL_MAP = {
    "BK": "Broken_Pipe",
    "CC": "Crack",
    "CL": "Crack",
    "LP": "Lateral_Protruding",
    "SD": "Surface_Damage",
    "IN": "Inside",
    "OUT": "Outside"
}


def get_prefix(filename):
    if "_" not in filename:
        return None
    return filename.split("_")[0].upper()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, default="sewer_full_raw")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        raise FileNotFoundError(f"Input folder not found: {input_dir}")

    prefix_counter = Counter()
    class_counter = Counter()
    unknown_files = []
    total_image_files = 0

    for path in input_dir.rglob("*"):
        if path.suffix.lower() not in IMAGE_EXTS:
            continue

        total_image_files += 1

        prefix = get_prefix(path.name)

        if prefix is None:
            unknown_files.append(path)
            continue

        prefix_counter[prefix] += 1

        if prefix in LABEL_MAP:
            class_counter[LABEL_MAP[prefix]] += 1
        else:
            unknown_files.append(path)

    print("\nTotal image files:", total_image_files)

    print("\nPrefix counts")
    for key, value in sorted(prefix_counter.items()):
        print(f"{key}: {value}")

    print("\nMerged class counts")
    for key, value in sorted(class_counter.items()):
        print(f"{key}: {value}")

    print("\nTotal valid images:", sum(class_counter.values()))
    print("Unknown or skipped images:", len(unknown_files))

    if len(unknown_files) > 0:
        print("\nExamples of unknown files")
        for path in unknown_files[:20]:
            print(path)


if __name__ == "__main__":
    main()