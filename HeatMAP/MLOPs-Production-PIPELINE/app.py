import os
import sys
import io
import numpy as np
import cv2
from datetime import datetime

from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from uvicorn import run as app_run

from magi_vision.constants import APP_HOST, APP_PORT
from magi_vision.logger import logging
from magi_vision.pipeline.prediction_pipeline import MAGIPredictionPipeline

# ============================================================
# MAGI VISION API
# ============================================================

app = FastAPI(
    title="MAGI Vision API",
    description=(
        "Plant health monitoring API for the MAGI quadruped robot. "
        "Supports single image prediction and canopy-level frame analysis."
    ),
    version="2.0.0",
)

# Templates
templates = Jinja2Templates(directory="templates")

# Initialize prediction pipeline (lazy load)
predictor = None


def get_predictor() -> MAGIPredictionPipeline:
    global predictor
    if predictor is None:
        predictor = MAGIPredictionPipeline()
    return predictor


# ============================================================
# API ROUTES
# ============================================================


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the MAGI Vision web interface."""
    return templates.TemplateResponse(request=request, name="magi.html")


@app.get("/health")
async def health_check():
    """API health check endpoint."""
    pred = get_predictor()
    return {
        "status": "healthy",
        "model_loaded": pred.model_loaded,
        "timestamp": datetime.now().isoformat(),
    }


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    """
    Predict plant health from a single uploaded image.

    Returns:
        JSON with class, confidence, health_score, class_probabilities
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file. Supported: JPG, PNG, BMP",
            )

        pred = get_predictor()
        result = pred.predict_single(image)

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/canopy")
async def predict_canopy(file: UploadFile = File(...)):
    """
    Canopy-level analysis of a camera frame.
    Splits into grid tiles, filters soil, classifies vegetation tiles.

    Returns:
        JSON with tile grid, vegetation coverage, mean health,
        stress distribution, and recommended Caspar action.
    """
    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(
                status_code=400,
                detail="Invalid image file.",
            )

        pred = get_predictor()
        result = pred.predict_canopy(frame)

        return JSONResponse(content=result)

    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Canopy analysis error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train")
async def trigger_training():
    """
    Trigger the full MAGI training pipeline.
    This is a long-running operation.
    """
    try:
        from magi_vision.pipeline.training_pipeline import MAGITrainPipeline

        logging.info("Training pipeline triggered via API")
        pipeline = MAGITrainPipeline()
        pipeline.run_pipeline()

        # Reload the predictor with the new model
        global predictor
        predictor = MAGIPredictionPipeline()

        return {
            "status": "success",
            "message": "Training pipeline completed. Model reloaded.",
        }

    except Exception as e:
        logging.error(f"Training error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/models")
async def list_models():
    """List deployed model versions."""
    from magi_vision.constants import TFLITE_EXPORT_DIR

    models = []
    if os.path.exists(TFLITE_EXPORT_DIR):
        for f in os.listdir(TFLITE_EXPORT_DIR):
            fpath = os.path.join(TFLITE_EXPORT_DIR, f)
            models.append({
                "file": f,
                "size_mb": round(os.path.getsize(fpath) / (1024 * 1024), 2),
                "modified": datetime.fromtimestamp(
                    os.path.getmtime(fpath)
                ).isoformat(),
            })
    return {"models": models}


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    app_run(app, host=APP_HOST, port=APP_PORT)