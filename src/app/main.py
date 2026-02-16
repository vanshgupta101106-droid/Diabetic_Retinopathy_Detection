from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import logging
import time
from typing import Dict

from src.app.schemas import PredictionResponse, HealthResponse
from src.app.predictor import DRPredictor

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global State for ML Model
models: Dict[str, DRPredictor] = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Load ML models on startup and clean up on shutdown.
    """
    logger.info("Loading DR Detection Model...")
    try:
        # Initialize the predictor (loads weights from disk)
        predictor = DRPredictor(model_name="efficientnet_b3") 
        models["predictor"] = predictor
        logger.info("Model loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise e
    
    yield
    
    # Clean up resources
    models.clear()
    logger.info("Model unloaded.")

# Create FastAPI App
app = FastAPI(
    title="Diabetic Retinopathy Detection API",
    description="API for detecting diabetic retinopathy from retinal fundus images.",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint to verify service status.
    """
    if "predictor" not in models:
        raise HTTPException(status_code=503, detail="Model not initialized")
    
    # We can add more checks here (GPU memory, etc.)
    return {
        "status": "healthy",
        "model_loaded": True,
        "version": "1.0.0"
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    """
    Predict DR grade from an uploaded image file.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    start_time = time.time()
    try:
        # Read file content
        contents = await file.read()
        
        # Get predictor instance
        predictor = models.get("predictor")
        if not predictor:
            raise HTTPException(status_code=503, detail="Model service unavailable.")
            
        # Run inference
        result = predictor.predict(contents)
        
        # Add metadata
        result["filename"] = file.filename
        result["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
        
        return result
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
