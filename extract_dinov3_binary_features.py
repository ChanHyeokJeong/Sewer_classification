import os
import argparse
import numpy as np
import torch

from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torchvision import datasets
from transformers import AutoImageProcessor, AutoModel


def load_dinov3(model_name, device):
    hf_token = os.environ.get("HF_TOKEN", None)

    if hf_token is not None:
        processor = AutoImageProcessor.from_pretrained(model_name, token=hf_token)
        model = AutoModel.from_pretrained(model_name, token=hf_token)
    else:
        processor = AutoImageProcessor.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)

    model = model.to(device)
    model.eval()

    return processor, model


def extract_features(data_dir, split_name, model_name, output_dir, batch_size):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    split_dir = os.path.join(data_dir, split_name)

    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Split folder not found: {split_dir}")

    dataset = datasets.ImageFolder(root=split_dir)
    class_names = dataset.classes

    processor, model = load_dinov3(model_name, device)

    image_paths = [sample[0] for sample in dataset.samples]
    image_labels = [sample[1] for sample in dataset.samples]

    features = []
    labels = []

    print()
    print("Split:", split_name)
    print("Number of images:", len(image_paths))
    print("Classes:", class_names)
    print("Device:", device)

    for start_idx in tqdm(range(0, len(image_paths), batch_size), desc=f"Extracting {split_name}"):
        batch_paths = image_paths[start_idx:start_idx + batch_size]
        batch_labels = image_labels[start_idx:start_idx + batch_size]

        images = []

        for path in batch_paths:
            image = Image.open(path).convert("RGB")
            images.append(image)

        inputs = processor(images=images, return_tensors="pt")
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)

        batch_features = outputs.pooler_output.detach().cpu().numpy()

        features.append(batch_features)
        labels.extend(batch_labels)

    features = np.vstack(features)
    labels = np.array(labels)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / f"X_{split_name}.npy", features)
    np.save(output_dir / f"y_{split_name}.npy", labels)

    with open(output_dir / "class_names.txt", "w", encoding="utf-8") as f:
        for class_name in class_names:
            f.write(class_name + "\n")

    print(f"Saved X_{split_name}.npy:", features.shape)
    print(f"Saved y_{split_name}.npy:", labels.shape)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="facebook/dinov3-vits16-pretrain-lvd1689m")
    parser.add_argument("--batch_size", type=int, default=32)

    args = parser.parse_args()

    for split_name in ["train", "val", "test"]:
        extract_features(
            data_dir=args.data_dir,
            split_name=split_name,
            model_name=args.model_name,
            output_dir=args.output_dir,
            batch_size=args.batch_size
        )

    print()
    print("DINOv3 feature extraction completed.")


if __name__ == "__main__":
    main()