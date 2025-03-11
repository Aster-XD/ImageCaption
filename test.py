import torch
from PIL import Image
import pytesseract
from models.blip import blip_decoder

# Set device and image size
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 384

# Function to load your demo image (ensure you have this defined)
def load_demo_image(image_path, image_size, device):
    from torchvision import transforms
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

# Path to your custom image
image_path = '/content/canva-DAGMyCgrDNg^UntitledDesign.png'

# Load and prepare image for BLIP and OCR
pil_image = Image.open(image_path).convert("RGB")
image_tensor = load_demo_image(image_path, image_size, device)

# Extract text using pytesseract
extracted_text = pytesseract.image_to_string(pil_image)
#print("Extracted text:", extracted_text)

# Load BLIP model
model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
model.eval()
model = model.to(device)

# Generate caption from BLIP model
with torch.no_grad():
    caption = model.generate(image_tensor, sample=False, num_beams=1, max_length=20, min_length=5)
    caption_text = caption[0]
    #print('BLIP Caption:', caption_text)

# Combine BLIP caption with extracted text (if any)
final_caption = caption_text
if extracted_text.strip():
    final_caption += " |  " + extracted_text.strip()

#print('Final Caption:', final_caption)


import google.generativeai as genai

genai.configure(api_key="AIzaSyCqMzs1Cu-hs3O6va_QmlkI3Qoo56lrnXw")


model = genai.GenerativeModel("gemini-2.0-flash")


response = model.generate_content(f"Write different types of caption for social media based on the given text which was generated from an image: {final_caption}")
print(response.text)


