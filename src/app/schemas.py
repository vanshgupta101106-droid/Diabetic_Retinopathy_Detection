"""
API Response Schemas
====================

Pydantic models that define the JSON response structure for the
``/`` (health-check) and ``/predict`` endpoints.
"""

from typing import Dict, Optional

from pydantic import BaseModel


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
    preprocessed_base64: Optional[str] = None
    stage_grade: Optional[int] = None
    risk_level: Optional[str] = None
    clinical_recommendation: Optional[str] = None
    clinical_notes: Optional[str] = None
