import os
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

model_name = "facebook/dinov3-vits16-pretrain-lvd1689m"
image_path = "test_image.png"

device = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(image_path):
    image = Image.new("RGB", (224, 224), color=(120, 120, 120))
    image.save(image_path)

processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model = model.to(device)
model.eval()

image = Image.open(image_path).convert("RGB")
inputs = processor(images=image, return_tensors="pt")
inputs = {k: v.to(device) for k, v in inputs.items()}

with torch.no_grad():
    outputs = model(**inputs)

embedding = outputs.pooler_output

print("Device =", device)
print("Embedding shape =", embedding.shape)
print("First five values =", embedding[0, :5])