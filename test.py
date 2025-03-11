from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
from PIL import Image
import pytesseract
import google.generativeai as genai
from torchvision import transforms
from models.blip import blip_decoder  # Ensure this is correctly imported

# Initialize Flask
app = Flask(__name__)
CORS(app)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
image_size = 384

# Function to load the image for BLIP processing
def load_demo_image(image_path, image_size, device):
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    return image_tensor

# Load BLIP Model
model_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/models/model_base_capfilt_large.pth'
model = blip_decoder(pretrained=model_url, image_size=image_size, vit='base')
model.eval().to(device)

# Configure Gemini API
genai.configure(api_key="API_KEY")  # Use env variable in production
gemini_model = genai.GenerativeModel("gemini-2.0-flash")

@app.route("/generate-captions", methods=["POST"])
def generate_captions():
    if "image" not in request.files:
        return jsonify({"error": "No image file uploaded"}), 400

    image_file = request.files["image"]
    pil_image = Image.open(image_file).convert("RGB")

    # Extract text using OCR
    extracted_text = pytesseract.image_to_string(pil_image).strip()

    # Generate caption using BLIP
    image_tensor = load_demo_image(image_file, image_size, device)
    with torch.no_grad():
        caption = model.generate(image_tensor, sample=False, num_beams=1, max_length=20, min_length=5)[0]

    # Combine BLIP caption and OCR text
    final_caption = caption if not extracted_text else f"{caption} | {extracted_text}"

    # Get social media captions from Gemini
    gemini_prompt = f"Write different types of captions for social media based on this image description: {final_caption}"
    response = gemini_model.generate_content(gemini_prompt)

    return jsonify({"blip_caption": caption, "ocr_text": extracted_text, "gemini_captions": response.text})

# Run the app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
