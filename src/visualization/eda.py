"""
Exploratory Data Analysis Helpers
=================================

High-level helpers that combine data loading and visualization for use
in Jupyter notebooks.  Each function encapsulates a complete analysis step
so that notebook cells remain short and readable.
"""

import base64
import io
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    f1_score,
)

from src.utils.logger import get_logger
from src.visualization.plots import plot_image_grid

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# NB01 helpers
# ---------------------------------------------------------------------------


def load_and_show_preprocessing(
    df: pd.DataFrame,
    preprocessor,
    n_samples: int = 6,
    random_state: int = 42,
    save_folder: str = "eda",
) -> list:
    """Load random samples and display *Raw vs Processed* image grids.

    Args:
        df: Manifest DataFrame with ``path``, ``eye``, ``level``, ``patient_id``.
        preprocessor: ImagePreprocessor instance.
        n_samples: Number of samples to display.
        random_state: Random seed for reproducibility.
        save_folder: Sub-folder under ``reports/figures/``.

    Returns:
        List of sample row-dicts (reusable by subsequent cells).
    """
    samples = df.sample(n_samples, random_state=random_state).to_dict("records")
    raw_imgs, proc_imgs, labels, titles = [], [], [], []

    for row in samples:
        img_bgr = cv2.imread(row["path"])
        img_rgb = (
            cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            if img_bgr is not None
            else np.zeros((512, 512, 3), dtype=np.uint8)
        )

        processed = preprocessor.preprocess(img_rgb, eye_side=row["eye"])
        raw_imgs.append(img_rgb)
        proc_imgs.append(processed)
        labels.append(row["level"])
        titles.append(f"Class {row['level']}\nID: {row['patient_id']} ({row['eye']})")

    logger.info("Displaying Raw Images...")
    plot_image_grid(
        raw_imgs, labels, titles=titles,
        grid_shape=(2, 3), figsize=(12, 10),
        title="Raw Images (Original)", save_folder=save_folder,
    )
    plt.show()

    logger.info("Displaying Processed Images (Graham's Method)...")
    plot_image_grid(
        proc_imgs, labels, titles=titles,
        grid_shape=(2, 3), figsize=(12, 10),
        title="Processed (Graham's Method + Standardized Orientation)",
        save_folder=save_folder,
    )
    plt.show()

    return samples


def create_vessel_overlay(
    processed_img: np.ndarray,
    vessel_mask: np.ndarray,
    overlay_color: Tuple[int, int, int] = (130, 0, 75),
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a coloured vessel highlight onto a processed fundus image.

    Args:
        processed_img: (H, W, 3) uint8 RGB image.
        vessel_mask: Binary mask (same H, W) from vessel extraction.
        overlay_color: RGB tint applied where vessels are detected.
        alpha: Blending weight for the overlay colour.

    Returns:
        Blended image (same shape as input).
    """
    overlay = processed_img.copy()
    mask_bool = vessel_mask > 0
    color = np.array(overlay_color, dtype=np.float32)
    overlay[mask_bool] = (
        (1 - alpha) * overlay[mask_bool] + alpha * color
    ).astype(np.uint8)
    return overlay


def show_vessel_density_report(
    samples: list,
    preprocessor,
    save_folder: str = "eda",
) -> pd.DataFrame:
    """Compute vessel density for sample images and display an overlay grid.

    Args:
        samples: List of row-dicts with ``path``, ``eye``, ``patient_id``, ``level``.
        preprocessor: ImagePreprocessor instance.
        save_folder: Sub-folder under ``reports/figures/``.

    Returns:
        DataFrame with per-image vessel density scores.
    """
    qc_results, qc_images, qc_titles, qc_labels = [], [], [], []

    for row in samples:
        img = preprocessor.load_image(row["path"])
        proc_img = preprocessor.preprocess(img, eye_side=row["eye"])

        density = preprocessor.validate_quality(proc_img)["vessel_density"]
        mask = preprocessor.extract_vessel_mask(proc_img)
        overlay = create_vessel_overlay(proc_img, mask)

        qc_results.append({
            "ID": row["patient_id"],
            "Level": row["level"],
            "Vessel Density": density,
        })
        qc_images.append(overlay)
        qc_labels.append(row["level"])
        qc_titles.append(f"ID: {row['patient_id']}\nDensity: {density:.3f}")

    qc_df = pd.DataFrame(qc_results)
    logger.info("Vessel Density Report:\n%s", qc_df.round(4).to_string(index=False))

    if qc_images:
        n = len(qc_images)
        cols = min(n, 3) if n <= 6 else 5
        rows = int(np.ceil(n / cols))
        plot_image_grid(
            qc_images, qc_labels, titles=qc_titles,
            grid_shape=(rows, cols), figsize=(4 * cols, 4 * rows + 2),
            title="Quality Check: Vessel Density & Segmentation Map",
            save_folder=save_folder,
        )
        plt.show()

    logger.info("Higher density indicates coherent vessel structure (good focus/contrast).")
    return qc_df


def show_augmentation_verification(
    dataset,
    sample_idx: int = 0,
    n_augments: int = 12,
    grid_shape: tuple = (4, 3),
    save_folder: str = "eda",
    patient_id: Optional[str] = None,
) -> None:
    """Display multiple augmented views of one sample with an orange reference circle.

    Args:
        dataset: RetinopathyDataset with augmentation transforms.
        sample_idx: Index of the sample to augment.
        n_augments: How many augmented versions to draw.
        grid_shape: (rows, cols) for the output grid.
        save_folder: Sub-folder under ``reports/figures/``.
        patient_id: Optional patient ID for the log message.
    """
    if patient_id:
        logger.info(f"Visualizing augmentations for Patient: {patient_id}")

    images, titles, labels = [], [], []

    for i in range(n_augments):
        data_item = dataset[sample_idx]
        img_tensor = data_item["image"]
        label = data_item["label"]

        display_img = img_tensor.permute(1, 2, 0).numpy()
        debug_img = (
            (display_img * 255).astype(np.uint8)
            if display_img.max() <= 1.0
            else display_img.copy()
        )

        h, w = debug_img.shape[:2]
        cv2.circle(debug_img, (w // 2, h // 2), int(min(h, w) * 0.48), (255, 165, 0), 3)

        images.append(debug_img)
        titles.append(f"Aug #{i + 1}")
        labels.append(label.item() if hasattr(label, "item") else int(label))

    logger.info("Plotting Verification Grid...")
    plot_image_grid(
        images, labels=labels, titles=titles,
        grid_shape=grid_shape, figsize=(12, 16),
        title="IsotropicAffine Verification (Orange Circle = Perfect Enclosure)",
        save_folder=save_folder,
    )
    plt.show()


# ---------------------------------------------------------------------------
# NB02 helpers
# ---------------------------------------------------------------------------


def load_and_show_quality_examples(
    df: pd.DataFrame,
    preprocessor,
    threshold: float,
    n_images: int = 10,
    quality_type: str = "low",
    save_folder: str = "qc",
) -> None:
    """Load and display a grid of low- or high-quality retinal images.

    Args:
        df: Manifest DataFrame with ``quality_score``, ``path``, ``eye``, ``level``.
        preprocessor: ImagePreprocessor instance.
        threshold: Quality-score cutoff separating low from high quality.
        n_images: Number of example images to show.
        quality_type: ``"low"`` (below threshold) or ``"high"`` (above median).
        save_folder: Sub-folder under ``reports/figures/``.
    """
    if quality_type == "low":
        subset = df[df["quality_score"] < threshold]
        suptitle = "Low Quality Examples (Candidates for Exclusion)"
    else:
        high_thresh = df["quality_score"].median()
        subset = df[df["quality_score"] > high_thresh]
        suptitle = "High Quality Examples (Clear Vessels & Good Contrast)"

    sample = subset.sample(min(n_images, len(subset)))
    logger.info(f"Displaying {quality_type} quality examples...")

    images, labels, titles = [], [], []
    for _, row in sample.iterrows():
        try:
            original = preprocessor.load_image(row["path"])
            processed = preprocessor.preprocess(original, eye_side=row["eye"])
            images.append(processed)
        except Exception:
            images.append(np.zeros((512, 512, 3), dtype=np.uint8))
        labels.append(row["level"])
        titles.append(f"Score: {row['quality_score']:.4f}\nLevel: {row['level']}")

    cols = 5
    rows = int(np.ceil(len(images) / cols))
    plot_image_grid(
        images, labels, titles=titles,
        grid_shape=(rows, cols), figsize=(20, 4 * rows),
        title=suptitle, save_folder=save_folder,
    )
    plt.show()


# ---------------------------------------------------------------------------
# NB03 helpers
# ---------------------------------------------------------------------------


def print_classification_summary(
    y_true,
    y_pred,
    class_names: Optional[List[str]] = None,
) -> None:
    """Print comprehensive multi-class classification metrics.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Optional display names for each class.
    """
    if class_names is None:
        class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

    acc = accuracy_score(y_true, y_pred)
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    kappa = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    logger.info(
        "\n" + "=" * 50
        + "\n--- Multi-Class Classification Analysis (5 Grades) ---"
        + "\n" + "=" * 50
        + f"\nOverall Accuracy: {acc:.4f}"
        + f"\nBalanced Accuracy: {balanced_acc:.4f}"
        + f"\nQuadratic Weighted Kappa (QWK): {kappa:.4f} (Key Clinical Metric)"
        + f"\nF1-Score (Macro): {f1_macro:.4f}"
        + f"\nF1-Score (Weighted): {f1_weighted:.4f}"
    )
    logger.info("\nPer-Class Metrics:\n%s", classification_report(y_true, y_pred, target_names=class_names))


def load_and_show_error_samples(
    results_df: pd.DataFrame,
    data_dir: Path,
    n_samples: int = 8,
    random_state: int = 42,
    model_name: str = "",
    save_folder: str = "evaluation",
) -> None:
    """Load misclassified images and display them in a grid.

    Args:
        results_df: Predictions DataFrame with ``image_name``, ``true``, ``pred``.
        data_dir: Root data directory containing ``train/`` subfolder.
        n_samples: Maximum number of errors to display.
        random_state: Random seed for sampling.
        model_name: Model identifier for the plot title.
        save_folder: Sub-folder under ``reports/figures/``.
    """
    if "image_name" not in results_df.columns:
        logger.warning("Column 'image_name' not found in predictions.")
        return

    errors = results_df[results_df["true"] != results_df["pred"]]
    if errors.empty:
        logger.info("Model has 100%% accuracy! No errors to visualize.")
        return

    sample_errors = errors.sample(n=min(n_samples, len(errors)), random_state=random_state)
    images, labels, preds, titles = [], [], [], []

    for _, row in sample_errors.iterrows():
        img_path = data_dir / "train" / f"{row['image_name']}.jpeg"
        try:
            img = cv2.imread(str(img_path))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img)
            labels.append(row["true"])
            preds.append(row["pred"])
            titles.append(f"True: {row['true']} | Pred: {row['pred']}")
        except Exception:
            pass

    if images:
        logger.info("Visualizing Error Cases...")
        plot_image_grid(
            images, labels, preds=preds, titles=titles,
            title=f"{model_name} Failure Analysis",
            save_folder=save_folder,
        )
        plt.show()


# ---------------------------------------------------------------------------
# NB04 helpers
# ---------------------------------------------------------------------------


def test_api_and_visualize(
    client,
    manifest_path: Path,
    min_level: int = 3,
) -> None:
    """Send a sample image to ``/predict`` and display the Grad-CAM result.

    Args:
        client: An initialised ``fastapi.testclient.TestClient``.
        manifest_path: Path to the dataset manifest CSV.
        min_level: Minimum DR severity level for sample selection.
    """
    from PIL import Image as PILImage  # deferred to avoid hard dependency

    class_map = {0: "No DR", 1: "Mild", 2: "Moderate", 3: "Severe", 4: "Proliferative"}

    sample_image_path = None
    true_label_desc = "Unknown"

    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        severe_df = df[df["level"] >= min_level]
        if not severe_df.empty:
            selected = severe_df.sample(1).iloc[0]
            sample_image_path = selected["path"]
            true_label_desc = f"{selected['level']} ({class_map.get(selected['level'], '?')})"
            logger.info(f"Selected sample: {sample_image_path} (Level: {true_label_desc})")

    if not sample_image_path or not Path(sample_image_path).exists():
        logger.warning("No suitable sample image found.")
        return

    logger.info("Sending POST request to /predict...")
    try:
        with open(sample_image_path, "rb") as f:
            response = client.post(
                "/predict",
                files={"file": ("sample.jpeg", f, "image/jpeg")},
            )
    except Exception as e:
        logger.error(f"Error during request: {e}")
        return

    if response.status_code != 200:
        logger.error(f"API Request Failed: {response.status_code}")
        logger.error(response.json())
        return

    result = response.json()
    logger.info(
        "-" * 50
        + "\nAPI RESPONSE SUCCESS"
        + f"\nFilename:  {result.get('filename')}"
        + f"\nDiagnosis: {result['diagnosis']}"
        + f"\nConfidence: {result['confidence']:.4f}"
        + f"\nReferral Advised: {result['referral_advised']}"
        + "\n" + "-" * 50
    )

    if result.get("gradcam_base64"):
        img_data = base64.b64decode(result["gradcam_base64"])
        gradcam_img = PILImage.open(io.BytesIO(img_data))
        original_img = PILImage.open(sample_image_path)

        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        axes[0].imshow(original_img)
        axes[0].set_title(f"Original Image\nTrue Label: {true_label_desc}")
        axes[0].axis("off")

        axes[1].imshow(gradcam_img)
        axes[1].set_title(
            f"API Prediction (Grad-CAM)\n{result['diagnosis']} ({result['confidence']:.2f})"
        )
        axes[1].axis("off")

        plt.tight_layout()
        plt.show()
