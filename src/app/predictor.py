"""
Prediction Service
==================

Thread-safe inference wrapper that loads model weights, runs preprocessing,
generates Grad-CAM overlays, and returns structured prediction results.
"""

import base64
import threading
from io import BytesIO
from typing import Any, Dict, Tuple, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from src.config import MODELS_DIR
from src.data.preprocessing import ImagePreprocessor
from src.models.explainability import GradCAM, get_target_layer
from src.models.model import DRModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DRPredictor:
    def __init__(self, model_name: str = "efficientnet_b3", device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Initializing Predictor on {self.device}...")

        # Grad-CAM hooks are stateful, so we serialise inference calls
        self.lock = threading.Lock()

        self.model = DRModel(model_name=model_name, num_classes=5, pretrained=False)

        weights_path = MODELS_DIR / f"{model_name}_best.pth"
        if not weights_path.exists():
            logger.warning(f"Weights not found at {weights_path}. Running with random weights.")
        else:
            logger.info(f"Loading weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state_dict)

        self.model.to(self.device)
        self.model.eval()

        self.preprocessor = ImagePreprocessor(target_size=(512, 512))
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        self.class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

        try:
            target_layer = get_target_layer(self.model, model_name)
            self.grad_cam = GradCAM(self.model, target_layer)
            logger.info(f"Grad-CAM initialized on layer: {target_layer}")
        except Exception as e:
            logger.warning(f"Failed to initialize Grad-CAM: {e}")
            self.grad_cam = None

    def _process_image(self, image_data: Union[bytes, np.ndarray]) -> Tuple[torch.Tensor, np.ndarray]:
        """Convert raw bytes or numpy array to a normalised tensor."""
        if isinstance(image_data, bytes):
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image bytes.")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_data, np.ndarray):
            img = image_data
        else:
            raise TypeError(f"Unsupported image type: {type(image_data)}")

        # Eye-side info isn't available via the API, so we skip orientation flipping
        processed = self.preprocessor.preprocess(img, enhance_contrast=True)

        tensor = torch.from_numpy(processed).permute(2, 0, 1).float() / 255.0
        tensor = self.normalize(tensor)
        return tensor.unsqueeze(0), processed

    def _generate_overlay(self, img: np.ndarray, heatmap: np.ndarray) -> str:
        """Blend a Grad-CAM heatmap onto *img* and return as a base64 PNG string."""
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_PLASMA)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
        overlay = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

        buffered = BytesIO()
        Image.fromarray(overlay).save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """Run end-to-end prediction: raw bytes -> JSON result dict.

        Note: ``@torch.no_grad()`` is intentionally omitted because Grad-CAM
        needs gradients for its backward pass.
        """
        tensor_tuple = self._process_image(image_bytes)
        tensor = tensor_tuple[0].to(self.device).requires_grad_()
        processed_img = tensor_tuple[1]

        heatmap = None
        probs = None
        diagnosis = "Unknown"
        referral_advised = False
        confidence = 0.0

        with self.lock:
            try:
                if self.grad_cam:
                    heatmap, class_idx, probs_tensor = self.grad_cam(tensor)
                    probs = probs_tensor.detach().cpu().numpy()[0]
                else:
                    with torch.no_grad():
                        logits = self.model(tensor)
                        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            except Exception as e:
                logger.error(f"Inference/Grad-CAM failed: {e}")
                if probs is None:
                    with torch.no_grad():
                        logits = self.model(tensor)
                        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        if probs is not None:
            top_class_idx = np.argmax(probs)
            confidence = float(probs[top_class_idx])
            diagnosis = self.class_names[top_class_idx]
            referral_advised = bool(top_class_idx >= 1)

        gradcam_b64 = None
        if heatmap is not None:
            try:
                gradcam_b64 = self._generate_overlay(processed_img, heatmap)
            except Exception as e:
                logger.error(f"Overlay generation failed: {e}")

        return {
            "diagnosis": diagnosis,
            "referral_advised": referral_advised,
            "confidence": confidence,
            "probabilities": {name: float(p) for name, p in zip(self.class_names, probs, strict=False)}
            if probs is not None
            else {},
            "gradcam_base64": gradcam_b64,
        }
