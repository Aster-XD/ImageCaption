import torch
from transformers import Blip2Processor, Blip2ForConditionalGeneration
from PIL import Image
import pytesseract
from fastapi import FastAPI, UploadFile, File
import google.generativeai as genai
import io

app = FastAPI()

# Load BLIP-2 Processor and Model
processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

def captionImage(image_path):
    image = Image.open(io.BytesIO(image_path))
    extracted_text = pytesseract.image_to_string(image)
    #return extracted_text.strip()

    image = image.convert("RGB")
    inputs = processor(images=image, return_tensors="pt").to(device, torch.float16)
    # Generate caption
    with torch.no_grad():
        caption_ids = model.generate(**inputs)
    caption = processor.batch_decode(caption_ids, skip_special_tokens=True)[0]
    final_caption = caption + " " + extracted_text
    return final_caption

@app.post("/process-image/")
async def process_image(file: UploadFile = File(...)):
    """API to process uploaded image and return captions + OCR text"""
    image = await file.read()
    caption = captionImage(image)
    genai.configure(api_key="AIzaSyCqMzs1Cu-hs3O6va_QmlkI3Qoo56lrnXw")
    model2 = genai.GenerativeModel("gemini-2.0-flash")
    response = model2.generate_content(f"Give me caption suggestions for social media based on the following text: {caption}")
    return {"caption": response.text}
