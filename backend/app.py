# backend/app.py
import io
import os
import sys
import traceback
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from model_loader import predict_image  # must return dict with keys "prediction","confidence"

app = FastAPI(title="Hand Gesture Backend")

# Allow any origin for local dev (adjust for prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

LOG_PATH = os.path.join(os.path.dirname(__file__), "error.log")


def log_exc(e: Exception):
    tb = traceback.format_exc()
    print(tb, file=sys.stderr, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write("\n\n--- Exception ---\n")
        f.write(tb)


@app.get("/")
def home():
    return {"status": "backend running"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    """
    Accepts multipart/form-data file=..., returns {"prediction": "...", "confidence": 0.9}
    """
    try:
        # read bytes and convert to PIL image
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")

        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # call synchronous predict function from model_loader
        result = predict_image(image)

        # validate result
        if isinstance(result, dict) and "prediction" in result:
            return result
        elif isinstance(result, (list, tuple)) and len(result) >= 2:
            return {"prediction": result[0], "confidence": float(result[1])}
        else:
            # Unknown return type
            return {"prediction": str(result), "confidence": 0.0}

    except HTTPException:
        raise
    except Exception as e:
        log_exc(e)
        raise HTTPException(status_code=500, detail="Server error — check backend/error.log")
