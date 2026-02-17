"""
Grad-CAM (Gradient-weighted Class Activation Mapping)
=====================================================

Produces visual explanations for CNN predictions by highlighting
which spatial regions most influenced the predicted class.
"""

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


class GradCAM:
    """Grad-CAM wrapper around a target convolutional layer.

    Args:
        model: The classification model.
        target_layer: The convolutional layer to hook into.
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.gradients: Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, module: nn.Module, input: Tuple, output: torch.Tensor) -> None:
        """Forward-hook: cache the layer's output activations."""
        self.activations = output

    def _save_gradient(self, module: nn.Module, grad_input: Tuple, grad_output: Tuple) -> None:
        """Backward-hook: cache the gradients flowing into the layer."""
        self.gradients = grad_output[0]

    def __call__(self, x: torch.Tensor, class_idx: Optional[int] = None) -> Tuple[np.ndarray, int, torch.Tensor]:
        """Generate a Grad-CAM heatmap for a single input image.

        Args:
            x: Input tensor of shape ``(1, C, H, W)``.
            class_idx: Target class index (default: argmax of logits).

        Returns:
            ``(heatmap, class_idx, softmax_probs)``
        """
        self.model.zero_grad()
        output = self.model(x)

        if class_idx is None:
            class_idx = torch.argmax(output).item()

        score = output[0, class_idx]
        score.backward()

        gradients = self.gradients
        activations = self.activations

        weights = torch.mean(gradients, dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * activations, dim=1, keepdim=True)
        cam = torch.relu(cam)
        cam = cam - torch.min(cam)
        cam = cam / (torch.max(cam) + 1e-7)

        return cam.cpu().detach().numpy()[0, 0], class_idx, torch.softmax(output, dim=1)


def get_target_layer(model: nn.Module, model_name: str) -> nn.Module:
    """Return the target convolutional layer for Grad-CAM.

    Args:
        model: A DRModel instance.
        model_name: Architecture name (``'resnet50'``, ``'efficientnet_b0'``, etc.).

    Returns:
        The target nn.Module layer.

    Raises:
        ValueError: If the architecture is not recognised.
    """
    if "resnet" in model_name:
        return model.backbone.layer4[-1]
    if "efficientnet" in model_name:
        return model.backbone.features[-1]
    if "densenet" in model_name:
        return model.backbone.features.denseblock4
    raise ValueError(f"Unknown architecture '{model_name}' for Grad-CAM mapping")
