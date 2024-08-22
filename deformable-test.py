from PIL import Image
import requests
from transformers import AutoImageProcessor, DeformableDetrForObjectDetection
import torch

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)
image

visual_encoder = DeformableDetrForObjectDetection.from_pretrained("SenseTime/deformable-detr")
visual_processor = AutoImageProcessor.from_pretrained("SenseTime/deformable-detr")

inputs = visual_processor(images=image, return_tensors="pt")

with torch.no_grad():
    outputs = visual_encoder(**inputs)

print(outputs['last_hidden_state'])