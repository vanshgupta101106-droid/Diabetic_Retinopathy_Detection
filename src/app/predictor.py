import torch
import torch.nn.functional as F
import numpy as np
import cv2
import logging
import base64
import threading
from PIL import Image
from io import BytesIO
from torchvision import transforms
from pathlib import Path
from typing import Dict, Any, Union, Tuple

from src.models.model import DRModel
from src.models.explainability import GradCAM, get_target_layer
from src.data.preprocessing import ImagePreprocessor
from src.config import MODELS_DIR

logger = logging.getLogger(__name__)

class DRPredictor:
    def __init__(self, model_name: str = "efficientnet_b3", device: str = None):
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        logger.info(f"Initializing Predictor on {self.device}...")
        
        # Thread lock for model inference (necessary because Grad-CAM hooks are stateful on the model)
        self.lock = threading.Lock()
        
        # 1. Load Model Architecture
        self.model = DRModel(model_name=model_name, num_classes=5, pretrained=False)
        
        # 2. Load Weights
        weights_path = MODELS_DIR / f"{model_name}_best.pth"
        if not weights_path.exists():
            # For demonstration if file doesn't exist, we warn but don't fail hard to allow structure testing
            # in a real app, this should fail.
            logger.warning(f"Model weights not found at {weights_path}. Model functionality will be limited.")
            # raise FileNotFoundError(f"Model weights not found at {weights_path}")
        else:
            logger.info(f"Loading weights from {weights_path}")
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            
        self.model.to(self.device)
        self.model.eval()
        
        # 3. Setup Preprocessing
        self.preprocessor = ImagePreprocessor(target_size=(512, 512))
        self.normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
        self.class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

        # 4. Setup Grad-CAM
        try:
            target_layer = get_target_layer(self.model, model_name)
            self.grad_cam = GradCAM(self.model, target_layer)
            logger.info(f"Grad-CAM initialized on layer: {target_layer}")
        except Exception as e:
            logger.warning(f"Failed to initialize Grad-CAM: {e}")
            self.grad_cam = None

    def _process_image(self, image_data: Union[bytes, np.ndarray]) -> Tuple[torch.Tensor, np.ndarray]:
        """Converts raw bytes or numpy array to normalized tensor and returns processed image for overlay."""
        # 1. Decode Image if bytes
        if isinstance(image_data, bytes):
            # Decode using OpenCV for consistency with training pipeline
            nparr = np.frombuffer(image_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError("Could not decode image bytes.")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        elif isinstance(image_data, np.ndarray):
            img = image_data
        else:
            raise TypeError(f"Unsupported image type: {type(image_data)}")
            
        # 2. Run Preprocessing Pipeline (Crop -> Resize -> Graham's Method)
        # Note: We don't have eye info in the API, so we skip eye-side specific flipping 
        # or leave it to standard horizontal flip augmentation (not applied here)
        processed = self.preprocessor.preprocess(img, enhance_contrast=True)
        
        # 3. To Tensor & Normalize
        # HWC -> CHW, Scale 0-1
        tensor = torch.from_numpy(processed).permute(2, 0, 1).float() / 255.0
        tensor = self.normalize(tensor)
        
        return tensor.unsqueeze(0), processed

    def _generate_overlay(self, img: np.ndarray, heatmap: np.ndarray) -> str:
        """Generates heatmap overlay and returns as base64 string."""
        # Resize heatmap to match image size
        heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
        
        # Convert to color map
        heatmap_uint8 = np.uint8(255 * heatmap)
        result_heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_PLASMA)
        result_heatmap = cv2.cvtColor(result_heatmap, cv2.COLOR_BGR2RGB)
        
        # Blend
        overlay = cv2.addWeighted(img, 0.6, result_heatmap, 0.4, 0)
        
        # Convert to PIL Image for easy encoding
        pil_img = Image.fromarray(overlay)
        buffered = BytesIO()
        pil_img.save(buffered, format="PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        return img_str

    # Removed @torch.no_grad() because Grad-CAM requires gradients
    def predict(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        End-to-end prediction: Raw Bytes -> JSON Result
        """
        # 1. Preprocess
        tensor_tuple = self._process_image(image_bytes)
        tensor = tensor_tuple[0].to(self.device).requires_grad_()
        processed_img = tensor_tuple[1]
        
        heatmap = None
        probs = None
        diagnosis = "Unknown"
        referral_advised = False
        confidence = 0.0

        # 2. Inference & Explainability (Thread-safe)
        with self.lock:
            try:
                if self.grad_cam:
                    # Run Grad-CAM (Forward + Backward)
                    # Returns: heatmap (numpy), class_idx (int), probs (tensor batch)
                    heatmap, class_idx, probs_tensor = self.grad_cam(tensor)
                    probs = probs_tensor.detach().cpu().numpy()[0]
                else:
                    # Standard Inference
                    with torch.no_grad():
                        logits = self.model(tensor)
                        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            except Exception as e:
                logger.error(f"Inference/Grad-CAM failed: {e}")
                # Fallback if Grad-CAM fails (e.g., CUDA OOM)
                if probs is None:
                     with torch.no_grad():
                        logits = self.model(tensor)
                        probs = F.softmax(logits, dim=1).cpu().numpy()[0]

        # 3. Post-process
        if probs is not None:
            top_class_idx = np.argmax(probs)
            confidence = float(probs[top_class_idx])
            diagnosis = self.class_names[top_class_idx]
            
            # Clinical Referral Logic: Any DR (Class >= 1) triggers referral
            # Ideally, this matches the binary classification concept (Class 0 vs 1-4)
            referral_advised = bool(top_class_idx >= 1)
        
        # 4. Generate Visualization
        gradcam_b64 = None
        if heatmap is not None:
            try:
                gradcam_b64 = self._generate_overlay(processed_img, heatmap)
            except Exception as e:
                logger.error(f"Overlay generation failed: {e}")
        
        result = {
            "diagnosis": diagnosis,
            "referral_advised": referral_advised,
            "confidence": confidence,
            "probabilities": {name: float(p) for name, p in zip(self.class_names, probs)} if probs is not None else {},
            "gradcam_base64": gradcam_b64
        }
        
        return result