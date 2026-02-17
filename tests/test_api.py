"""
Tests for FastAPI Application
==============================

Integration tests for the health-check and prediction endpoints
using mocked model inference.
"""

from unittest.mock import MagicMock

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.app.main import app, models


@pytest.fixture
def mock_predictor():
    predictor = MagicMock()
    predictor.predict.return_value = {
        "diagnosis": "No DR",
        "referral_advised": False,
        "confidence": 0.99,
        "probabilities": {"No DR": 0.99, "Mild": 0.01},
        "gradcam_base64": "base64placeholder",
        "processing_time_ms": 120.5,
    }
    return predictor


@pytest.fixture
def client(mock_predictor, monkeypatch):
    # Patch DRPredictor so lifespan uses our mock instead of loading real weights
    monkeypatch.setattr(
        "src.app.main.DRPredictor",
        lambda *args, **kwargs: mock_predictor,
    )

    with TestClient(app) as c:
        yield c

    models.clear()


def test_health_check(client):
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"
    assert response.json()["model_loaded"] is True


def test_predict_endpoint_success(client, mock_predictor):
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    _, encoded_img = cv2.imencode(".jpg", img)

    response = client.post("/predict", files={"file": ("test.jpg", encoded_img.tobytes(), "image/jpeg")})

    assert response.status_code == 200
    data = response.json()
    assert data["diagnosis"] == "No DR"
    assert "gradcam_base64" in data
    mock_predictor.predict.assert_called_once()


def test_predict_invalid_file_type(client):
    response = client.post("/predict", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert response.status_code == 400
