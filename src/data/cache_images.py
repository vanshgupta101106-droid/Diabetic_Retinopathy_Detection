"""
Cache Images
============

Pre-processes the entire dataset (Graham's method) and saves 512x512 images
to disk so the training loop can skip heavy CPU preprocessing.

Usage::

    python -m src.data.cache_images
"""

from pathlib import Path

import cv2
import pandas as pd
from tqdm import tqdm

from src.config import DATA_DIR, PROCESSED_DATA_DIR
from src.data.make_dataset import create_manifest
from src.data.preprocessing import ImagePreprocessor
from src.utils.logger import get_logger

logger = get_logger(__name__)


def cache_dataset(target_size=(512, 512)):
    """Pre-process and cache all images to disk for faster training.

    Args:
        target_size: ``(H, W)`` output resolution.
    """
    input_manifest = PROCESSED_DATA_DIR / "manifest.csv"
    output_dir = PROCESSED_DATA_DIR / "cached_images"
    output_manifest = PROCESSED_DATA_DIR / "manifest_cached.csv"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not input_manifest.exists():
        logger.info("Manifest not found, creating it first...")
        df = create_manifest(DATA_DIR, PROCESSED_DATA_DIR)
        df.to_csv(input_manifest, index=False)
    else:
        df = pd.read_csv(input_manifest)

    logger.info(f"Loaded manifest with {len(df)} images.")
    preprocessor = ImagePreprocessor(target_size=target_size)

    new_paths = []
    success_count = 0

    for _idx, row in tqdm(df.iterrows(), total=len(df), desc="Caching Images"):
        original_path = Path(row["path"])
        save_path = output_dir / original_path.name

        if save_path.exists():
            new_paths.append(str(save_path))
            continue

        try:
            img = preprocessor.load_image(str(original_path))
            img = preprocessor.preprocess(img, eye_side=row.get("eye"))
            cv2.imwrite(str(save_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
            new_paths.append(str(save_path))
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to process {original_path}: {e}")
            new_paths.append(str(original_path))

    df["path"] = new_paths
    df.to_csv(output_manifest, index=False)
    logger.info(f"Caching complete. Processed {success_count} new images.")
    logger.info(f"Saved cached manifest to {output_manifest}")


if __name__ == "__main__":
    cache_dataset()
