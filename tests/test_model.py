import torch
import pytest
from src.models.model import DRModel

@pytest.fixture
def model():
    return DRModel(model_name='resnet50', num_classes=5, pretrained=False)

def test_model_forward_pass(model):
    """Test that model accepts tensor and returns logits"""
    # Create dummy tensor (Batch size 1, 3 channels, 512x512)
    x = torch.randn(1, 3, 512, 512)
    
    with torch.no_grad():
        output = model(x)
        
    assert output.shape == (1, 5)

def test_efficientnet_head_replacement():
    """Test that EfficientNet loads and replaces the head correctly"""
    model = DRModel(model_name='efficientnet_b3', num_classes=5, pretrained=False)
    # Check if classifier layer is correct size
    # EfficientNet B3 classifier output is 1536 features
    assert model.model_name == 'efficientnet_b3'
    # The last layer should be replaced with Linear(features, 5)
    # The torchvision efficientnet structure is .classifier[1]
    assert model.backbone.classifier[1].out_features == 5

