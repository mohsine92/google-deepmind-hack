from fastapi import APIRouter, UploadFile, File
import numpy as np
import cv2
from PIL import Image
import io
from backend.app.ml.train import model

router = APIRouter()

@router.post("/analyze")
async def analyze_image(file: UploadFile = File(...)):

    # 1. Lire l'image envoyée
    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    image = np.array(image)

    # 2. Conversion OpenCV
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # 3. Preprocessing (base microscopy)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # 4. Features simples mais utiles scientifiquement
    features = {
        "mean_intensity": float(np.mean(gray)),
        "std_intensity": float(np.std(gray)),
        "min_intensity": float(np.min(gray)),
        "max_intensity": float(np.max(gray)),
        "blur_intensity": float(np.mean(blurred))
    }

    ml_result = model.predict(features)

    return {
        "filename": file.filename,
        "features": features,
        "ml_result": ml_result
    }