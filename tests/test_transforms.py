"""Tests for src.data.transforms."""

import albumentations as A
import numpy as np
import torch

from src.data.transforms import IsotropicAffineTransform, get_transforms


def _make_synthetic_image(h=512, w=512):
    """Return a random uint8 RGB image."""
    return np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)


class TestGetTransforms:
    def test_train_returns_compose(self):
        t = get_transforms("train")
        assert isinstance(t, A.Compose)

    def test_val_returns_compose(self):
        t = get_transforms("val")
        assert isinstance(t, A.Compose)

    def test_train_output_is_tensor(self):
        t = get_transforms("train")
        img = _make_synthetic_image()
        result = t(image=img)["image"]
        assert isinstance(result, torch.Tensor)

    def test_val_output_is_tensor(self):
        t = get_transforms("val")
        img = _make_synthetic_image()
        result = t(image=img)["image"]
        assert isinstance(result, torch.Tensor)

    def test_train_output_shape(self):
        t = get_transforms("train")
        img = _make_synthetic_image(256, 256)
        result = t(image=img)["image"]
        assert result.shape == (3, 256, 256)

    def test_val_output_shape(self):
        t = get_transforms("val")
        img = _make_synthetic_image(256, 256)
        result = t(image=img)["image"]
        assert result.shape == (3, 256, 256)


class TestIsotropicAffineTransform:
    def test_output_shape_preserved(self):
        t = IsotropicAffineTransform(p=1.0)
        img = _make_synthetic_image(128, 128)
        out = t.apply(img)
        assert out.shape == img.shape

    def test_identity_when_disabled(self):
        """With p=0 the transform should not be applied."""
        pipeline = A.Compose([IsotropicAffineTransform(p=0.0)])
        img = _make_synthetic_image(64, 64)
        result = pipeline(image=img)["image"]
        np.testing.assert_array_equal(result, img)
