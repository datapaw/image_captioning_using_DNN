from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch

# Load a pre-trained model and processor from Hugging Face
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Load your image
image_path = './dataset/Images/23445819_3a458716c1.jpg'
image = Image.open(image_path)

# Process the image and generate a caption
inputs = processor(image, return_tensors="pt")
output = model.generate(**inputs)

caption = processor.decode(output[0], skip_special_tokens=True)
print("Caption:", caption)
