"""
Image Preprocessing
===================

Core image processing for retinal fundus images using Ben Graham's method
(circle crop, resize, illumination correction) and vessel-based quality metrics.
"""

from typing import Any, Dict, Tuple

import cv2
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ImagePreprocessor:
    """
    Handles preprocessing for Retinal Fundus Images.
    Key steps:
    1. Border removal (Center Crop to Fundus)
    2. Resizing
    3. Orientation Standardization (Flip Left to Right)
    4. Illumination Correction (Graham's Method)
    5. Quality Assessment (Blur, Brightness)
    """

    def __init__(self, target_size: Tuple[int, int] = (512, 512), tolerance: int = 7, sigmaX: int = 30):
        """
        Args:
            target_size: (height, width) to resize to.
            tolerance: Tolerance for black pixel detection (0-255).
            sigmaX: Gaussian blur sigma for Graham's method.
        """
        self.target_size = target_size
        self.tolerance = tolerance
        self.sigmaX = sigmaX

    def load_image(self, path: str) -> np.ndarray:
        """Load an RGB image from disk."""
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Image not found at {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def crop_black_borders_fast(self, img: np.ndarray) -> np.ndarray:
        """Crop to the bounding box of the largest contour (the fundus)."""
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        _, mask = cv2.threshold(gray, self.tolerance, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return img

        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return img[y : y + h, x : x + w]

    def preprocess(self, img: np.ndarray, eye_side: str = None, enhance_contrast: bool = True) -> np.ndarray:
        """Full pipeline: crop -> resize -> orient -> Graham's method -> circular mask."""
        cropped = self.crop_black_borders_fast(img)
        resized = self.resize_with_padding(cropped)

        # Build a padding mask to prevent black borders turning gray during Graham's method
        mask = None
        if enhance_contrast:
            dummy_mask = np.full(cropped.shape, 255, dtype=np.uint8)
            mask = self.resize_with_padding(dummy_mask)

        # Flip left eyes so the optic disc is always on the right
        if eye_side and eye_side.lower().strip() == "left":
            resized = cv2.flip(resized, 1)
            if mask is not None:
                mask = cv2.flip(mask, 1)

        if enhance_contrast:
            resized = self.adjust_illumination_graham(resized)
            if mask is not None:
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                resized = cv2.bitwise_and(resized, mask)

        return self.apply_circular_mask(resized)

    def adjust_illumination_graham(self, img: np.ndarray) -> np.ndarray:
        """Ben Graham's illumination correction: ``4*img - 4*blur + 128``.

        Highlights vessels and lesions by subtracting the low-frequency background.
        """
        blurred = cv2.GaussianBlur(img, (0, 0), self.sigmaX)
        return cv2.addWeighted(img, 4, blurred, -4, 128)

    def apply_circular_mask(self, img: np.ndarray, radius_ratio: float = 0.98) -> np.ndarray:
        """Apply a circular mask to remove gray halos left by Graham's method."""
        h, w = img.shape[:2]
        center = (int(w / 2), int(h / 2))
        radius = int(min(h, w) / 2 * radius_ratio)

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(mask, center, radius, 255, -1)
        return cv2.bitwise_and(img, img, mask=mask)

    def resize_with_padding(self, img: np.ndarray) -> np.ndarray:
        """Resize to ``target_size`` preserving aspect ratio with black padding."""
        h, w = img.shape[:2]
        target_h, target_w = self.target_size

        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        canvas = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        pad_top = (target_h - new_h) // 2
        pad_left = (target_w - new_w) // 2
        canvas[pad_top : pad_top + new_h, pad_left : pad_left + new_w] = resized
        return canvas

    def validate_quality(self, img: np.ndarray) -> Dict[str, Any]:
        """Compute vessel-density quality metric.

        Sharp images have well-defined vessel trees (high density), while
        blurry or dark images lose fine details (low density).
        """
        vessel_mask = self.extract_vessel_mask(img)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        valid_area = cv2.countNonZero(gray)
        vessel_area = cv2.countNonZero(vessel_mask)
        density = (vessel_area / valid_area) if valid_area > 0 else 0.0
        return {"vessel_density": float(density)}

    def extract_vessel_mask(self, img: np.ndarray) -> np.ndarray:
        """Extract a binary vessel mask via green-channel adaptive thresholding."""
        green = img[:, :, 1] if len(img.shape) == 3 else img

        # Mask out the black background
        _, fundus_mask = cv2.threshold(green, 1, 255, cv2.THRESH_BINARY)

        # Adaptive threshold picks pixels darker than their 15x15 neighborhood (C=6)
        vessels = cv2.adaptiveThreshold(green, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 6)
        vessels = cv2.bitwise_and(vessels, fundus_mask)

        # Remove single-pixel noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
        return cv2.morphologyEx(vessels, cv2.MORPH_OPEN, kernel)
