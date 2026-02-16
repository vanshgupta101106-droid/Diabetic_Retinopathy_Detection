from pydantic import BaseModel
from typing import Dict, Optional

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str

class PredictionResponse(BaseModel):
    filename: Optional[str] = None
    diagnosis: str
    referral_advised: bool
    confidence: float
    probabilities: Dict[str, float]
    processing_time_ms: Optional[float] = None
    gradcam_base64: Optional[str] = None
