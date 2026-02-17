"""
Model Architecture
==================

Wraps torchvision backbones (ResNet, EfficientNet, DenseNet) and swaps
the classification head for DR severity grading.
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    DenseNet121_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B3_Weights,
    ResNet50_Weights,
)


class DRModel(nn.Module):
    """Configurable image classifier for diabetic retinopathy grading.

    Args:
        model_name: Backbone identifier
            (``'resnet50'``, ``'efficientnet_b0'``, ``'efficientnet_b3'``, ``'densenet121'``).
        num_classes: Number of output classes (default 5 for DR severity).
        pretrained: Whether to initialise with ImageNet weights.
    """

    def __init__(self, model_name: str = "resnet50", num_classes: int = 5, pretrained: bool = True):
        super().__init__()
        self.model_name = model_name.lower()
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.backbone = self._get_backbone()

    def _get_backbone(self) -> nn.Module:
        """Instantiate the backbone and replace the classification head."""
        weights = None
        if self.pretrained:
            if self.model_name == "resnet50":
                weights = ResNet50_Weights.DEFAULT
            elif self.model_name == "efficientnet_b0":
                weights = EfficientNet_B0_Weights.DEFAULT
            elif self.model_name == "efficientnet_b3":
                weights = EfficientNet_B3_Weights.DEFAULT
            elif self.model_name == "densenet121":
                weights = DenseNet121_Weights.DEFAULT

        # ResNet50
        if self.model_name == "resnet50":
            model = models.resnet50(weights=weights)
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, self.num_classes)
            return model

        # EfficientNet (B0 / B3)
        if self.model_name.startswith("efficientnet"):
            if self.model_name == "efficientnet_b0":
                model = models.efficientnet_b0(weights=weights)
            elif self.model_name == "efficientnet_b3":
                model = models.efficientnet_b3(weights=weights)
            else:
                raise ValueError(f"EfficientNet variant {self.model_name} not expressly supported yet.")
            in_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(in_features, self.num_classes)
            return model

        # DenseNet121
        if self.model_name == "densenet121":
            model = models.densenet121(weights=weights)
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, self.num_classes)
            return model

        raise ValueError(f"Model {self.model_name} not supported/implemented yet.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the backbone.

        Args:
            x: Input tensor of shape ``(B, 3, H, W)``.

        Returns:
            Logits tensor of shape ``(B, num_classes)``.
        """
        return self.backbone(x)

    def get_input_size(self) -> tuple[int, int]:
        """Return the expected input spatial size ``(H, W)``."""
        return (512, 512)
