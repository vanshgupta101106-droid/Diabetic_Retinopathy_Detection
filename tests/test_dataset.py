"""
Tests for RetinopathyDataset
=============================

Verifies dataset initialization, indexing, and error handling
using synthetic images written to a temporary directory.
"""

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.data.dataset import RetinopathyDataset
from src.data.transforms import get_transforms


@pytest.fixture
def sample_df(tmp_path):
    """Create a minimal manifest with synthetic JPEG images on disk."""
    records = []
    for i in range(4):
        img = np.random.randint(50, 200, (64, 64, 3), dtype=np.uint8)
        cv2.circle(img, (32, 32), 28, (255, 255, 255), -1)
        path = tmp_path / f"{i}_left.jpeg"
        cv2.imwrite(str(path), img)
        records.append({
            "image": f"{i}_left",
            "path": str(path),
            "level": i % 5,
            "eye": "left",
            "patient_id": str(i),
        })
    return pd.DataFrame(records)


def test_dataset_length(sample_df):
    ds = RetinopathyDataset(sample_df)
    assert len(ds) == len(sample_df)


def test_getitem_returns_expected_keys(sample_df):
    ds = RetinopathyDataset(sample_df)
    item = ds[0]
    assert "image" in item and "label" in item
    assert isinstance(item["image"], torch.Tensor)
    assert isinstance(item["label"], torch.Tensor)


def test_image_tensor_shape(sample_df):
    ds = RetinopathyDataset(sample_df)
    item = ds[0]
    assert item["image"].shape[0] == 3  # CHW


def test_label_dtype(sample_df):
    ds = RetinopathyDataset(sample_df)
    item = ds[0]
    assert item["label"].dtype == torch.long


def test_with_transforms(sample_df):
    transforms = get_transforms("val", (512, 512))
    ds = RetinopathyDataset(sample_df, transforms=transforms)
    item = ds[0]
    assert isinstance(item["image"], torch.Tensor)
    assert item["image"].shape[0] == 3


def test_return_metadata(sample_df):
    ds = RetinopathyDataset(sample_df, return_metadata=True)
    item = ds[0]
    assert "id" in item
    assert "eye" in item
    assert "path" in item


def test_missing_image_returns_black(sample_df):
    """Dataset should not crash on missing files; returns a black placeholder."""
    sample_df.at[0, "path"] = "/nonexistent/path.jpeg"
    ds = RetinopathyDataset(sample_df)
    item = ds[0]
    assert item["image"].shape[0] == 3
