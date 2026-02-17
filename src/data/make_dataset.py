"""
Dataset Creation
================

Scans raw data directories, parses filenames and labels, and writes the
master manifest CSV consumed by the training pipeline.
"""

from pathlib import Path

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


def create_manifest(data_dir: Path, output_path: Path) -> pd.DataFrame:
    """Scan ``data_dir/train/`` and join with ``trainLabels.csv`` to build a manifest."""
    labels_csv = data_dir / "trainLabels.csv"
    if not labels_csv.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_csv}")

    df = pd.read_csv(labels_csv)
    img_dir = data_dir / "train"

    # image column contains e.g. "10_left" → path = train/10_left.jpeg
    df["path"] = df["image"].apply(lambda x: str(img_dir / (x + ".jpeg")))
    df["patient_id"] = df["image"].apply(lambda x: x.split("_")[0])
    df["eye"] = df["image"].apply(lambda x: x.split("_")[1])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Manifest saved to {output_path} with {len(df)} records.")
    return df


def load_filtered_dataset(manifest_path: Path, min_quality_score: float = 0.0) -> pd.DataFrame:
    """Load the manifest and optionally filter by quality score.

    Args:
        manifest_path: Path to the CSV manifest.
        min_quality_score: Minimum vessel density score (0.0 = include all).

    Returns:
        Filtered DataFrame.
    """
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")

    df = pd.read_csv(manifest_path)

    # Cached manifests may lack quality_score; merge from the main manifest
    if "quality_score" not in df.columns and min_quality_score > 0.0:
        main_manifest_path = manifest_path.parent / "manifest.csv"
        if main_manifest_path.exists():
            logger.info(f"Merging quality scores from {main_manifest_path.name}...")
            main_df = pd.read_csv(main_manifest_path)
            if "quality_score" in main_df.columns:
                scores = main_df[["image", "quality_score"]]
                df = df.merge(scores, on="image", how="left")

    if "quality_score" in df.columns and min_quality_score > 0.0:
        initial_count = len(df)
        df = df[df["quality_score"] >= min_quality_score].reset_index(drop=True)
        dropped = initial_count - len(df)
        if dropped > 0:
            logger.info(f"Quality Filter: Dropped {dropped} images below {min_quality_score}. Remaining: {len(df)}")
    elif min_quality_score > 0.0:
        logger.warning(f"'quality_score' column not found in {manifest_path.name} — skipping filter.")

    return df
