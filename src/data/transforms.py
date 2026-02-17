"""
Augmentation Pipelines
======================

Defines Albumentations-based augmentation pipelines for training and
validation stages.

Classes:
    IsotropicAffineTransform: Custom isotropic affine (used in notebooks).
"""

import albumentations as A
import cv2
import numpy as np
from albumentations.pytorch import ToTensorV2


class IsotropicAffineTransform(A.ImageOnlyTransform):
    """Isotropic affine transform that preserves circles.

    Uses internal numpy randomisation to regenerate parameters per image
    regardless of the pipeline's caching behaviour.

    Args:
        scale_limit: Max fractional scaling offset from 1.0.
        shift_limit: Max fractional translation.
        rotate_limit: Max rotation in degrees.
    """

    def __init__(
        self,
        scale_limit: float = 0.05,
        shift_limit: float = 0.05,
        rotate_limit: float = 360,
        always_apply: bool = False,
        p: float = 0.5,
    ):
        super().__init__(always_apply=always_apply, p=p)
        self.scale_limit = scale_limit
        self.shift_limit = shift_limit
        self.rotate_limit = rotate_limit

    def apply(self, image: np.ndarray, **params) -> np.ndarray:
        """Apply the transformation."""
        h, w = image.shape[:2]
        center = (w // 2, h // 2)

        angle = np.random.uniform(-self.rotate_limit, self.rotate_limit)
        scale = np.random.uniform(1 - self.scale_limit, 1 + self.scale_limit)
        dx = np.random.uniform(-self.shift_limit, self.shift_limit)
        dy = np.random.uniform(-self.shift_limit, self.shift_limit)

        tx = round(dx * w)
        ty = round(dy * h)

        M = cv2.getRotationMatrix2D(center, angle, scale)
        M[0, 2] += tx
        M[1, 2] += ty

        return cv2.warpAffine(
            image,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def get_transform_init_args_names(self):
        return ("scale_limit", "shift_limit", "rotate_limit")


def get_transforms(stage: str = "train", target_size: tuple = (512, 512)) -> A.Compose:
    """Return an Albumentations pipeline for a given stage.

    Args:
        stage: ``'train'`` or ``'val'``/``'test'``.
        target_size: ``(H, W)`` — unused (images are pre-sized), kept for API compat.

    Returns:
        Albumentations Compose pipeline.
    """
    if stage == "train":
        return A.Compose(
            [
                A.ShiftScaleRotate(
                    shift_limit=0.15,
                    scale_limit=0.2,
                    rotate_limit=180,
                    border_mode=cv2.BORDER_CONSTANT,
                    value=0,
                    p=0.8,
                ),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(
                    brightness_limit=0.1,
                    contrast_limit=0.1,
                    p=0.5,
                ),
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )
    else:
        return A.Compose(
            [
                A.Normalize(
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
                ToTensorV2(),
            ]
        )
