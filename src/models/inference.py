"""
Inference Pipeline
==================

Connects preprocessing, model prediction, and Grad-CAM visualization
into a single ``predict_and_visualize`` function.
"""

from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import cv2
import numpy as np
import requests
import torch
from PIL import Image
from torchvision import transforms

from src.data.preprocessing import ImagePreprocessor
from src.models.explainability import GradCAM
from src.utils.logger import get_logger
from src.visualization.plots import plot_inference_result

logger = get_logger(__name__)


def predict_and_visualize(
    image_path_or_url: str,
    model: torch.nn.Module,
    preprocessor: ImagePreprocessor,
    grad_cam: GradCAM,
    device: torch.device,
    eye_side: str = "left",
    true_label: Optional[str] = None,
    patient_info: Optional[Dict[str, Any]] = None,
    save_folder: str = "inference_demo",
) -> None:
    """Run the full inference pipeline: Load -> Preprocess -> Predict -> Explain -> Visualize.

    Args:
        image_path_or_url: Local file path or HTTP(S) URL to a fundus image.
        model: Loaded classification model.
        preprocessor: ImagePreprocessor instance.
        grad_cam: Initialised GradCAM wrapper.
        device: Torch device.
        eye_side: ``'left'`` or ``'right'``.
        true_label: Optional ground-truth label for annotation.
        patient_info: Optional dict of patient metadata for the plot header.
        save_folder: Sub-folder under ``reports/figures/`` to save the result.
    """
    try:
        # Load image from URL or local path
        if str(image_path_or_url).startswith("http"):
            response = requests.get(image_path_or_url)
            img_pil = Image.open(BytesIO(response.content))
            img = np.array(img_pil)
            if img.shape[-1] == 4:
                img = img[..., :3]
        else:
            path_obj = Path(image_path_or_url)
            if not path_obj.exists():
                raise FileNotFoundError(f"Image not found at {path_obj}")
            img = cv2.imread(str(path_obj))
            if img is None:
                raise ValueError(f"Failed to load image from {path_obj}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        processed = preprocessor.preprocess(img, eye_side=eye_side)
        vessel_mask = preprocessor.extract_vessel_mask(processed)
        quality = preprocessor.validate_quality(processed)

        tensor = torch.from_numpy(processed).permute(2, 0, 1).float() / 255.0
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        tensor = normalize(tensor).unsqueeze(0).to(device)
        tensor.requires_grad = True

        heatmap, class_idx, probs = grad_cam(tensor)

        class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
        pred_label = class_names[class_idx]
        confidence = probs[0, class_idx].item()
        prob_values = probs[0].detach().cpu().numpy()
        q_score = quality["vessel_density"] if isinstance(quality, dict) else quality

        plot_inference_result(
            original_img=img,
            processed_img=processed,
            heatmap=heatmap,
            class_name=pred_label,
            confidence=confidence,
            probs=prob_values,
            class_names=class_names,
            true_label=true_label,
            save_folder=save_folder,
            vessel_mask=vessel_mask,
            quality_score=q_score,
            patient_info=patient_info,
        )

    except Exception as e:
        logger.exception(f"Error in inference pipeline: {e}")
