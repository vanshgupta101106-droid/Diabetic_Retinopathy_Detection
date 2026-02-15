"""
Diabetic Retinopathy Detection - Quality Control Script
=======================================================

This script iterates through the entire dataset manifest, computes the 
vessel density quality score for each image, and updates the manifest.

It uses multiprocessing to speed up the computation across available CPU cores.
"""

import sys
import os
from pathlib import Path
import pandas as pd
import numpy as np
from tqdm import tqdm
import cv2
import argparse
from concurrent.futures import ProcessPoolExecutor

# Ensure project root is in path
current_file = Path(__file__).resolve()
project_root = current_file.parent.parent.parent
# Insert project root at the BEGINNING of sys.path to prioritize local modules
# over installed packages (especially since 'src' is a common generic name)
sys.path.insert(0, str(project_root))

try:
    from src.config import PROCESSED_DATA_DIR, DATA_DIR
    from src.data.preprocessing import ImagePreprocessor
except ImportError:
    # Double check if path insertion failed or different structure
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.config import PROCESSED_DATA_DIR, DATA_DIR
    from src.data.preprocessing import ImagePreprocessor

def process_single_image(args):
    """
    Worker function to process a single image.
    Args:
        args: Tuple of (path_str, eye_side_str)
    Returns:
        float: Vessel density score
    """
    path_str, eye_side = args
    
    try:
        # Initialize Preprocessor locally for each process
        # This avoids pickling issues and race conditions with OpenCV
        preprocessor = ImagePreprocessor(target_size=(512, 512), sigmaX=30)
        
        # Determine actual file path
        # Check if path in manifest is relative or absolute
        if os.path.isabs(path_str):
            img_path = Path(path_str)
        else:
            # Assume relative to project root or data dir
            img_path = project_root / path_str

        if not img_path.exists():
            return 0.0 # Penalty for missing file

        # Load & Preprocess
        # Note: Preprocessor.load_image expects string
        img = preprocessor.load_image(str(img_path))
        if img is None:
            return 0.0

        processed = preprocessor.preprocess(img, eye_side=eye_side)
        
        # Compute Quality
        quality = preprocessor.validate_quality(processed)
        return quality['vessel_density']
        
    except Exception as e:
        # Silently fail for the progress bar's sake, return bad score
        # In a production script, logging to a separate file would be better
        return 0.0

def assess_manifest_quality(input_manifest_path: Path, output_manifest_path: Path):
    """
    Computes quality scores for all images in the manifest using multiprocessing.
    """
    if not input_manifest_path.exists():
        print(f"Manifest not found at {input_manifest_path}")
        return

    df = pd.read_csv(input_manifest_path)
    print(f"Loaded manifest with {len(df)} images.")
    
    # Prepare arguments for parallel processing
    # Create list of tuples: [(path1, eye1), (path2, eye2), ...]
    tasks = list(zip(df['path'], df['eye']))
    
    # Determine workers (leave 1 core free)
    import multiprocessing
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Computing Vessel Density Scores using {num_workers} workers...")
    
    results = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # returns results in order
        results = list(tqdm(executor.map(process_single_image, tasks), total=len(tasks)))
            
    # Add to DataFrame
    df['quality_score'] = results
    
    # Save
    df.to_csv(output_manifest_path, index=False)
    print(f"Updated manifest saved to {output_manifest_path}")

    # Stats
    print("\n--- Quality Statistics ---")
    print(df['quality_score'].describe())
    
    # Suggest Threshold (e.g., 5th percentile)
    threshold = df['quality_score'].quantile(0.05)
    print(f"\nSuggested exclusion threshold (5th percentile): {threshold:.4f}")

if __name__ == "__main__":
    # Windows-safe execution block
    manifest_path = PROCESSED_DATA_DIR / "manifest.csv"
    assess_manifest_quality(manifest_path, manifest_path)
