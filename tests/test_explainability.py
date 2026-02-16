"""Tests for src.models.explainability."""

import torch
import pytest

from src.models.model import DRModel
from src.models.explainability import GradCAM, get_target_layer


@pytest.fixture
def resnet_model():
    return DRModel(model_name="resnet50", num_classes=5, pretrained=False)


class TestGetTargetLayer:
    def test_resnet(self, resnet_model):
        layer = get_target_layer(resnet_model, "resnet50")
        assert layer is resnet_model.backbone.layer4[-1]

    def test_efficientnet_b0(self):
        model = DRModel(model_name="efficientnet_b0", num_classes=5, pretrained=False)
        layer = get_target_layer(model, "efficientnet_b0")
        assert layer is model.backbone.features[-1]

    def test_densenet(self):
        model = DRModel(model_name="densenet121", num_classes=5, pretrained=False)
        layer = get_target_layer(model, "densenet121")
        assert layer is model.backbone.features.denseblock4

    def test_unknown_raises(self, resnet_model):
        with pytest.raises(ValueError):
            get_target_layer(resnet_model, "vgg19")


class TestGradCAM:
    def test_init_registers_hooks(self, resnet_model):
        layer = get_target_layer(resnet_model, "resnet50")
        cam = GradCAM(resnet_model, layer)
        assert cam.gradients is None
        assert cam.activations is None

    def test_forward_produces_heatmap(self, resnet_model):
        resnet_model.eval()
        layer = get_target_layer(resnet_model, "resnet50")
        cam = GradCAM(resnet_model, layer)

        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        heatmap, class_idx, probs = cam(x)

        assert heatmap.ndim == 2
        assert 0 <= class_idx <= 4
        assert probs.shape == (1, 5)
        assert abs(probs.sum().item() - 1.0) < 1e-5
