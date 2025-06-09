# backend.py
import google.generativeai as genai
from PIL import Image
from ultralytics import YOLO
from io import BytesIO

# Configuration: place best.pt, then set your Gemini credentials here once
WEIGHTS_PATH = 'best.pt'
GEMINI_API_URL = 'https://api.generative.google/v1beta2/models/gemini-2.0/flash:generate'
GEMINI_API_KEY = 'AIzaSyCl-Ys-YuIoTBzN0fI8gcLmyIZsRp_zxWY'  # <-- Replace with your key once

# Load YOLOv8 model
_model = YOLO(WEIGHTS_PATH)
_model.to('cpu')

import numpy as np

def detect_pest(image: Image.Image, imgsz: int = 640):
    """
    Run YOLOv8 inference on a PIL image and return annotated frame and pest name.
    """
    # Convert to numpy array (Ultralytics accepts np.ndarray)
    image_np = np.array(image.convert('RGB'))

    results = _model.predict(source=image_np, imgsz=imgsz, device='cpu')
    frame = results[0].plot()  # Annotated BGR image
    boxes = results[0].boxes
    if not boxes:
        return frame, None

    confs = boxes.conf.cpu().numpy()
    classes = boxes.cls.cpu().numpy().astype(int)
    best_idx = confs.argmax()
    pest_name = results[0].names[classes[best_idx]]
    return frame, pest_name

def get_advice(pest_name: str) -> dict:
    """
    Retrieve pest treatment advice via Gemini 2.0 Flash using google.generativeai SDK.
    """
    prompt = (
        f"Act as an expert crop protection advisor. For the pest named '{pest_name}', "
        f"provide a JSON response with the following keys:\n\n"
        f"- \"pest\": Scientific name of the pest\n"
        f"- \"common_name_en\": Common name in English\n"
        f"- \"common_name_hi\": Local name in Hindi (Devanagari script)\n"
        f"- \"chemical\": Name of an effective chemical pesticide (with application dosage and method in 1–2 lines)\n"
        f"- \"organic\": Name of an organic or natural remedy (with preparation or application instructions in 1–2 lines)\n"
        f"- \"prevention\": A short preventive practice farmers can follow to avoid this pest (1–2 lines only)\n"
        f"- \"crop_stage\": The stage of crop growth when this pest usually attacks (e.g., seedling, flowering)\n\n"
        f"Ensure the response is strictly valid JSON with only these keys and concise, actionable content—no extra commentary."
    )

    model = genai.GenerativeModel(model_name="gemini-2.0-flash")
    response = model.generate_content(prompt)
    
    # Try extracting JSON from response
    import json, re
    try:
        json_str = re.search(r'\{.*\}', response.text, re.DOTALL).group(0)
        return json.loads(json_str)
    except Exception as e:
        return {"error": f"Failed to parse Gemini response: {e}", "raw": response.text}
