"""
Tests for DRModel
==================

Verifies model initialization, forward pass, and head replacement
for all supported backbone architectures.
"""

import torch
import pytest
from src.models.model import DRModel


@pytest.fixture
def model():
    return DRModel(model_name='resnet50', num_classes=5, pretrained=False)


def test_model_forward_pass(model):
    """Test that model accepts tensor and returns logits"""
    x = torch.randn(1, 3, 512, 512)
    with torch.no_grad():
        output = model(x)
    assert output.shape == (1, 5)


@pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0", "efficientnet_b3", "densenet121"])
def test_all_supported_architectures(arch):
    """Each supported architecture should initialise and produce correct output shape."""
    model = DRModel(model_name=arch, num_classes=5, pretrained=False)
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = model(x)
    assert out.shape == (1, 5)


def test_efficientnet_head_replacement():
    """Test that EfficientNet loads and replaces the head correctly"""
    model = DRModel(model_name='efficientnet_b3', num_classes=5, pretrained=False)
    assert model.model_name == 'efficientnet_b3'
    assert model.backbone.classifier[1].out_features == 5


def test_unsupported_model_raises():
    """Unknown architecture names should raise ValueError."""
    with pytest.raises(ValueError):
        DRModel(model_name='vgg19', num_classes=5, pretrained=False)

