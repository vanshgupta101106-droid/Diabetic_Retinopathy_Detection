"""
FastAPI Application
===================

Defines the REST API for diabetic retinopathy detection.  The ``/predict``
endpoint accepts a fundus image upload, runs inference through
DRPredictor, and returns a JSON response
with diagnosis, confidence, and an optional Grad-CAM overlay.
"""

import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from src.app.predictor import DRPredictor
from src.app.schemas import HealthResponse, PredictionResponse
from src.config import PROJECT_ROOT
from src.utils.logger import get_logger

logger = get_logger(__name__)

models: Dict[str, DRPredictor] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load ML models on startup and clean up on shutdown.
    """
    logger.info("Loading DR Detection Model...")
    try:
        predictor = DRPredictor(model_name="efficientnet_b3")
        models["predictor"] = predictor
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e

    yield

    models.clear()
    logger.info("Model unloaded.")


# Create FastAPI App
app = FastAPI(
    title="Diabetic Retinopathy Detection API",
    description="API for detecting diabetic retinopathy from retinal fundus images with Grad-CAM explainability.",
    version="1.0.0",
    lifespan=lifespan,
)

# Mount sample images and reports if available
misc_dir = PROJECT_ROOT / "misc"
if misc_dir.exists():
    app.mount("/static/misc", StaticFiles(directory=str(misc_dir)), name="misc")

reports_dir = PROJECT_ROOT / "reports" / "figures" / "test_outputs"
if reports_dir.exists():
    app.mount("/static/reports", StaticFiles(directory=str(reports_dir)), name="reports")

TEMPLATE_PATH = Path(__file__).resolve().parent / "templates" / "index.html"


@app.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify service status.
    """
    if "predictor" not in models:
        raise HTTPException(status_code=503, detail="Model not initialized")

    return {"status": "healthy", "model_loaded": True, "version": "1.0.0"}


@app.get("/ui", response_class=HTMLResponse)
async def web_ui():
    """
    Interactive web interface for testing the model with retinal fundus images.
    """
    if TEMPLATE_PATH.exists():
        return HTMLResponse(content=TEMPLATE_PATH.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>RetinaAI Web UI</h1><p>Template not found.</p>")


@app.get("/api/training-history")
async def training_history():
    """
    Return training loss, accuracy, and Quadratic Weighted Kappa history.
    """
    history_file = PROJECT_ROOT / "reports" / "history_efficientnet_b3.joblib"
    if history_file.exists():
        try:
            import joblib

            data = joblib.load(history_file)
            return {k: [float(x) for x in v] for k, v in data.items()}
        except Exception as e:
            logger.warning(f"Failed to load training history: {e}")
    return {}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict DR grade from an uploaded image file.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    start_time = time.time()
    try:
        contents = await file.read()
        predictor = models.get("predictor")
        if not predictor:
            raise HTTPException(status_code=503, detail="Model service unavailable.")

        result = predictor.predict(contents, filename=file.filename)
        result["filename"] = file.filename
        result["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)

        return result

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e)) from e


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

