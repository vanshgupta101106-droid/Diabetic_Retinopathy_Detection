"""
Quality Control Script
======================

Iterates through the dataset manifest, computes vessel-density quality
scores for each image, and updates the manifest CSV.

Uses ProcessPoolExecutor for parallelism.
"""

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import DATA_DIR, PROCESSED_DATA_DIR
from src.data.preprocessing import ImagePreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Resolve project root for relative path handling inside workers
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def process_single_image(args: tuple) -> float:
    """Worker function — process one image and return its vessel density.

    Args:
        args: ``(path_str, eye_side)`` tuple.

    Returns:
        Vessel density score (0.0 on failure).
    """
    path_str, eye_side = args

    try:
        preprocessor = ImagePreprocessor(target_size=(512, 512), sigmaX=30)

        if os.path.isabs(path_str):
            img_path = Path(path_str)
        else:
            img_path = _PROJECT_ROOT / path_str

        if not img_path.exists():
            return 0.0

        img = preprocessor.load_image(str(img_path))
        if img is None:
            return 0.0

        processed = preprocessor.preprocess(img, eye_side=eye_side)
        quality = preprocessor.validate_quality(processed)
        return quality['vessel_density']

    except Exception:
        return 0.0


def assess_manifest_quality(input_manifest_path: Path, output_manifest_path: Path) -> None:
    """Compute quality scores for every image in the manifest.

    Args:
        input_manifest_path: Path to the source manifest CSV.
        output_manifest_path: Path to write the updated manifest CSV.
    """
    if not input_manifest_path.exists():
        logger.error(f"Manifest not found at {input_manifest_path}")
        return

    df = pd.read_csv(input_manifest_path)
    logger.info(f"Loaded manifest with {len(df)} images.")

    tasks = list(zip(df['path'], df['eye']))
    num_workers = max(1, multiprocessing.cpu_count() - 1)
    logger.info(f"Computing Vessel Density Scores using {num_workers} workers...")

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        results = list(tqdm(executor.map(process_single_image, tasks), total=len(tasks)))

    df['quality_score'] = results
    df.to_csv(output_manifest_path, index=False)
    logger.info(f"Updated manifest saved to {output_manifest_path}")

    logger.info("--- Quality Statistics ---")
    logger.info(f"\n{df['quality_score'].describe()}")

    threshold = df['quality_score'].quantile(0.05)
    logger.info(f"Suggested exclusion threshold (5th percentile): {threshold:.4f}")


if __name__ == "__main__":
    manifest_path = PROCESSED_DATA_DIR / "manifest.csv"
    assess_manifest_quality(manifest_path, manifest_path)
