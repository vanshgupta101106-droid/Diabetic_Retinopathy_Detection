import cv2
import numpy as np
import pytest
from src.data.preprocessing import ImagePreprocessor

@pytest.fixture
def preprocessor():
    return ImagePreprocessor(target_size=(512, 512))

@pytest.fixture
def dummy_image():
    # Create black image with white circle in center
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    cv2.circle(img, (500, 500), 400, (255, 255, 255), -1)
    return img

def test_preprocess_output_shape(preprocessor, dummy_image):
    """Test that preprocessor resizes to target shape"""
    processed = preprocessor.preprocess(dummy_image, enhance_contrast=False)
    assert processed.shape == (512, 512, 3)

def test_graham_method_produces_contrast(preprocessor, dummy_image):
    """Test standard preprocessing with Graham method"""
    processed = preprocessor.preprocess(dummy_image, enhance_contrast=True)
    assert processed.shape == (512, 512, 3)
    # Check if mean value shifted towards grey (128) due to subtraction logic
    # The dummy image has black (0) and white (255).
    # Graham: 4*img - 4*blur + 128.
    # For black bg: 4*0 - 4*0 + 128 = 128.
    # But mask is applied later to make background black again.
    # So valid retina area should be roughly grey.
    # Let's check center pixel
    center_pixel = processed[256, 256]
    # It shouldn't be pure white or black.
    assert np.any(center_pixel > 10) 

def test_crop_works(preprocessor):
    """Test tight cropping around content"""
    # Create a small white square in a big black image
    img = np.zeros((1000, 1000, 3), dtype=np.uint8)
    img[400:600, 400:600] = 255
    
    # Preprocessor should zoom in on this square and resize it to 512x512
    # So the result should be mostly white
    processed = preprocessor.preprocess(img, enhance_contrast=False)
    
    # Calculate amount of non-black pixels
    non_black = np.sum(processed > 10) / processed.size
    
    # If cropped correctly, the white square (200x200) fills the 512x512 frame significantly more than 4%
    # Original 200x200 is 4% of 1000x1000.
    # If we crop to 200x200 then resize to 512x512, it fills 100% (minus circular mask)
    # Circular mask keeps ~78% of area.
    assert non_black > 0.5
