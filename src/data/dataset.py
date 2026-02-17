"""
PyTorch Dataset
===============

Loads retinal fundus images, applies Graham's preprocessing on-the-fly,
and feeds augmented tensors to the training loop.
"""

from typing import Callable, Dict, Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.data.preprocessing import ImagePreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RetinopathyDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        preprocessor: Optional[ImagePreprocessor] = None,
        transforms: Optional[Callable] = None,
        return_metadata: bool = False,
    ):
        """
        Args:
            df: Pandas DataFrame containing 'path', 'level', 'eye' columns.
            preprocessor: Instance of ImagePreprocessor for core normalization (Crop/Graham's).
            transforms: Albumentations pipeline for Data Augmentation.
            return_metadata: If True, returns patient_id and eye_side (useful for inference/error analysis).
        """
        self.df = df.reset_index(drop=True)
        self.preprocessor = preprocessor
        self.transforms = transforms
        self.return_metadata = return_metadata

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        row = self.df.iloc[idx]
        image_path = str(row["path"])
        label = row["level"]
        eye_side = row["eye"]

        if self.preprocessor:
            try:
                img = self.preprocessor.load_image(image_path)
                img = self.preprocessor.preprocess(img, eye_side=eye_side)
            except Exception as e:
                logger.warning(f"Failed to load/process {image_path}: {e}")
                target_h, target_w = self.preprocessor.target_size
                img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        else:
            img = cv2.imread(image_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img = np.zeros((512, 512, 3), dtype=np.uint8)

        if self.transforms:
            try:
                img = self.transforms(image=img)["image"]
            except Exception as e:
                logger.warning(f"Transform error on {image_path}: {e}")

        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0

        sample = {"image": img, "label": torch.tensor(label, dtype=torch.long)}

        if self.return_metadata:
            sample["id"] = str(row.get("patient_id", idx))
            sample["eye"] = str(eye_side)
            sample["path"] = image_path

        return sample
