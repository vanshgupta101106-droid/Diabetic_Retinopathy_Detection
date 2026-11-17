"""
Visualization
=============

Unified plotting interface with a Deep-Purple-to-Orange colour theme
for consistent, publication-ready figures.
"""

from typing import Any, Dict, List, Optional

import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from cycler import cycler
from matplotlib.colors import LinearSegmentedColormap
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src.config import REPORTS_DIR
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Project theme palette — ordered from Healthy (purple) to Severe (orange).
THEME_COLORS = ["#4B0082", "#762B98", "#A157AE", "#CC82C4", "#FF8C00"]


def _save_plot(fig, title: str, save_folder: str = None):
    """
    Internal helper to save figures to reports/figures/{save_folder}.
    """
    if not save_folder:
        output_dir = REPORTS_DIR / "figures"
    else:
        output_dir = REPORTS_DIR / "figures" / save_folder

    output_dir.mkdir(parents=True, exist_ok=True)

    # Clean title for filename
    clean_title = title.lower().replace(" ", "_").replace(":", "").replace("-", "_").replace("\n", "_")
    save_path = output_dir / f"{clean_title}.png"

    try:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Saved plot to {save_path}")
    except Exception as e:
        logger.error(f"Failed to save plot: {e}")


def set_plot_style():
    """Set the project-wide matplotlib and seaborn style.

    Palette runs from Deep Purple (No DR) to Dark Orange (Proliferative).
    """
    plt.style.use("ggplot")

    selected_colors = ["#4B0082", "#762B98", "#A157AE", "#CC82C4", "#FF8C00"]

    mpl.rcParams["axes.prop_cycle"] = cycler("color", selected_colors)
    sns.set_palette(sns.color_palette(selected_colors))

    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["axes.titlesize"] = 14
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["lines.linewidth"] = 2.0
    mpl.rcParams["grid.alpha"] = 0.5
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False

    return selected_colors


def plot_class_distribution(
    df: pd.DataFrame, label_col: str = "level", title: str = "Class Epidemiology", save_folder: str = "eda"
):
    """
    Plots the count of each DR severity level.
    """
    colors = set_plot_style()

    fig, ax = plt.subplots(figsize=(10, 6))

    # Count plot
    sns.countplot(x=label_col, data=df, hue=label_col, palette=colors, legend=False, ax=ax)

    # Add percentages
    total = len(df)
    for p in ax.patches:
        percentage = f"{100 * p.get_height() / total:.1f}%"
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha="center", va="bottom")

    ax.set_title(title)
    ax.set_xlabel("DR Severity Level (0-4)")
    ax.set_ylabel("Patient Count")

    plt.tight_layout()
    _save_plot(fig, title, save_folder)
    return fig


def plot_image_grid(
    images: List[np.ndarray],
    labels: List[int],
    preds: Optional[List[int]] = None,
    titles: Optional[List[str]] = None,
    grid_shape: tuple = (2, 4),
    figsize: tuple = (16, 8),
    title: str = "Sample Retinal Fundus Images",
    save_folder: str = "eda",
):
    """
    Plots a grid of images with labels and colored borders.
    """
    colors = set_plot_style()
    rows, cols = grid_shape
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()

    for i, ax in enumerate(axes):
        if i < len(images):
            # Force aspect='equal' to prevent any display distortion
            ax.imshow(images[i], aspect="equal")

            # Determine class color
            label_val = labels[i]
            try:
                c_idx = int(label_val)
                c_idx = max(0, min(c_idx, 4))
                class_color = colors[c_idx]
            except (ValueError, IndexError):
                class_color = "black"

            if titles:
                ax_title = titles[i]
            else:
                ax_title = f"Grade: {labels[i]}"
                if preds is not None:
                    ax_title += f"\npred: {preds[i]}"

            ax.set_title(ax_title, fontsize=12, color=class_color, fontweight="bold")

            # Add colored border (Frame)
            # Turn spines on explicitly for the frame effect
            for side in ["top", "bottom", "left", "right"]:
                ax.spines[side].set_visible(True)
                ax.spines[side].set_color(class_color)
                ax.spines[side].set_linewidth(4)

            # Remove ticks but keep the colored frame
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)  # Disable grid inside image

        else:
            ax.axis("off")

    plt.suptitle(title, fontsize=16, y=0.99)
    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.5, w_pad=0.5)
    _save_plot(fig, title, save_folder)
    return fig


def plot_learning_curves(history: Dict[str, List[float]], save_folder: str = "evaluation", title_prefix: str = ""):
    """
    Plots Training and Validation Loss/Accuracy/Kappa.

    Args:
        history (dict): Dictionary with lists for 'train_loss', 'val_loss', 'train_kappa', 'val_kappa', etc.
    """
    set_plot_style()

    # Identify available metrics
    metrics = [k for k in history.keys() if k.startswith("train_")]

    # Check for standalone validation metrics (like Kappa) that don't have a training counterpart
    standalone_val = [k for k in history.keys() if k.startswith("val_") and f"train_{k[4:]}" not in metrics]

    all_metrics_to_plot = metrics + standalone_val
    n_plots = len(all_metrics_to_plot)

    if n_plots == 0:
        return

    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]

    epochs = range(1, len(history[list(history.keys())[0]]) + 1)

    # Colors
    col_train = "#4B0082"  # Deep Purple
    col_val = "#FF8C00"  # Dark Orange

    for ax, metric in zip(axes, all_metrics_to_plot, strict=False):
        if metric.startswith("train_"):
            metric_name = metric.replace("train_", "")
            val_metric = f"val_{metric_name}"

            ax.plot(epochs, history[metric], label=f"Train {metric_name}", color=col_train, linewidth=2)
            if val_metric in history:
                ax.plot(
                    epochs,
                    history[val_metric],
                    label=f"Val {metric_name}",
                    color=col_val,
                    linewidth=2,
                    linestyle="--",
                )
        else:
            # Standalone validation metric
            metric_name = metric.replace("val_", "")
            ax.plot(epochs, history[metric], label=f"Val {metric_name}", color=col_val, linewidth=2, linestyle="-")

        ax.set_title(f"{metric_name.capitalize()} Evolution")
        ax.set_xlabel("Epochs")
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_plot(fig, f"{title_prefix}Learning_Curves", save_folder)
    return fig


def _add_heatmap_annotations(ax, data, fmt="d", threshold=0.3, fontsize=12):
    """Manually add text annotations to heatmap cells.

    Args:
        ax: Matplotlib axes containing the heatmap.
        data: 2-D array of numeric values to annotate.
        fmt: Format string (e.g. ``"d"`` for int) or an array of pre-formatted strings.
        threshold: Ratio above which text color switches to white.
        fontsize: Font size for annotation text.
    """
    rows, cols = data.shape
    max_val = np.max(data) if np.max(data) > 0 else 1

    annot_array = None
    if isinstance(fmt, (np.ndarray, list)):
        annot_array = fmt
        fmt = ""

    for i in range(rows):
        for j in range(cols):
            val = data[i, j]
            text_val = f"{val:{fmt}}" if annot_array is None else str(annot_array[i, j])

            ratio = val / max_val
            color = "white" if ratio > threshold else "black"

            ax.text(
                j + 0.5,
                i + 0.5,
                text_val,
                ha="center",
                va="center",
                color=color,
                fontweight="bold",
                fontsize=fontsize,
            )


def plot_model_diagnostics(
    results_df: pd.DataFrame, n_classes: int = 5, save_folder: str = "evaluation", title_prefix: str = ""
):
    """
    Generates a diagnostic panel:
    1. Confusion Matrix
    2. ROC Curves (One-vs-Rest)
    3. Per-Class F1/Accuracy (Bar Plot)
    """
    set_plot_style()
    colors = sns.color_palette()

    fig = plt.figure(figsize=(18, 6), constrained_layout=True)
    gs = fig.add_gridspec(1, 3)

    cmap_purple = LinearSegmentedColormap.from_list("custom_purple", ["#fafafa", "#4B0082"])

    ax_cm = fig.add_subplot(gs[0])
    cm = confusion_matrix(results_df["true"], results_df["pred"], labels=list(range(n_classes)))
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    sns.heatmap(
        cm_norm,
        annot=False,
        cmap=cmap_purple,
        ax=ax_cm,
        cbar=False,
        xticklabels=[str(i) for i in range(n_classes)],
        yticklabels=[str(i) for i in range(n_classes)],
    )

    _add_heatmap_annotations(ax_cm, cm, fmt="d", threshold=0.3)

    acc = accuracy_score(results_df["true"], results_df["pred"])
    ax_cm.set_title(f"Confusion Matrix (Acc: {acc:.1%})")
    ax_cm.set_ylabel("True Label")
    ax_cm.set_xlabel("Predicted Label")

    ax_roc = fig.add_subplot(gs[1])
    prob_cols = [c for c in results_df.columns if c.startswith("prob_")]
    if len(prob_cols) == n_classes:
        y_score = results_df[prob_cols].values

        # Binarize labels
        from sklearn.preprocessing import label_binarize

        y_test = label_binarize(results_df["true"], classes=range(n_classes))

        for i in range(n_classes):
            fpr, tpr, _ = roc_curve(y_test[:, i], y_score[:, i])
            roc_auc = auc(fpr, tpr)
            ax_roc.plot(fpr, tpr, label=f"Class {i} (AUC = {roc_auc:.2f})", lw=2)

        ax_roc.plot([0, 1], [0, 1], "k--", lw=1)
        ax_roc.set_title("ROC Curves (One-vs-Rest)")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right")
    else:
        ax_roc.text(0.5, 0.5, "Probability data missing\nfor ROC", ha="center")

    ax_bar = fig.add_subplot(gs[2])
    cm_diag = cm.diagonal()
    cm_sum = cm.sum(axis=1)
    recall = np.divide(cm_diag, cm_sum, out=np.zeros_like(cm_diag, dtype=float), where=cm_sum != 0)

    x_classes = np.arange(n_classes)
    sns.barplot(x=x_classes, y=recall, hue=x_classes, ax=ax_bar, palette=colors, legend=False)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xlabel("Class")
    ax_bar.set_ylabel("Recall (Sensitivity)")
    ax_bar.set_title("Per-Class Recall")

    for i, v in enumerate(recall):
        ax_bar.text(i, v + 0.02, f"{v:.2f}", ha="center")

    _save_plot(fig, f"{title_prefix}Model_Diagnostics", save_folder)
    return fig


def plot_confidence_abstention_panel(
    results_df: pd.DataFrame, confidence_threshold: float = 0.7, save_folder: str = "evaluation", title_prefix: str = ""
):
    """
    Visualizes confidence distribution and impact of abstention.
    Adapts logic for Multi-class: Confidence is max(probability vector).
    """
    set_plot_style()

    prob_cols = [c for c in results_df.columns if c.startswith("prob_")]
    if not prob_cols:
        return

    probs = results_df[prob_cols].values
    confidence = np.max(probs, axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax_hist = axes[0]
    sns.histplot(confidence, bins=20, ax=ax_hist, kde=True, color="#4B0082")
    ax_hist.axvline(confidence_threshold, color="#FF8C00", linestyle="--", label=f"Threshold {confidence_threshold}")
    ax_hist.set_title("Prediction Confidence Distribution")
    ax_hist.set_xlabel("Max Probability (Confidence)")
    ax_hist.legend()

    ax_curve = axes[1]
    thresholds = np.linspace(0.0, 0.95, 20)
    accuracies = []
    retentions = []

    true_labels = results_df["true"].values
    pred_labels = results_df["pred"].values

    for t in thresholds:
        mask = confidence >= t
        if mask.sum() > 0:
            acc = accuracy_score(true_labels[mask], pred_labels[mask])
            retention = mask.mean()
        else:
            acc = np.nan
            retention = 0.0
        accuracies.append(acc)
        retentions.append(retention)

    ax_curve.plot(thresholds, accuracies, label="Accuracy on Retained", color="#4B0082", marker="o")
    ax_curve.set_xlabel("Confidence Threshold")
    ax_curve.set_ylabel("Accuracy")

    ax2 = ax_curve.twinx()
    ax2.plot(thresholds, retentions, label="Retention Rate", color="#FF8C00", linestyle="--", marker="x")
    ax2.set_ylabel("Retention Rate")

    lines, labels = ax_curve.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax_curve.legend(lines + lines2, labels + labels2, loc="center right")

    ax_curve.set_title("Reliability Curve (Accuracy vs Retention)")

    plt.tight_layout()
    _save_plot(fig, f"{title_prefix}Confidence_Analysis", save_folder)
    return fig


def plot_inference_result(
    original_img: np.ndarray,
    processed_img: np.ndarray,
    heatmap: np.ndarray,
    class_name: str,
    confidence: float,
    probs: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    true_label: Optional[str] = None,
    save_folder: str = "inference",
    vessel_mask: Optional[np.ndarray] = None,
    quality_score: Optional[Any] = None,
    patient_info: Optional[Dict[str, Any]] = None,
):
    """
    Visualizes inference result: Original, Processed, and Grad-CAM (Plasma).
    """
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05])

    if patient_info:
        info_str = " | ".join([f"{k}: {v}" for k, v in patient_info.items()])
        fig.suptitle(info_str, fontsize=16, y=0.98, fontweight="bold")

    ax_orig = fig.add_subplot(gs[0, 0])
    ax_proc = fig.add_subplot(gs[0, 1])
    ax_cam = fig.add_subplot(gs[0, 2])
    ax_probs = fig.add_subplot(gs[1, :3])

    # Ensure images are uint8
    if original_img.dtype != np.uint8:
        if original_img.max() <= 1.0:
            original_img = (original_img * 255).astype(np.uint8)
        else:
            original_img = original_img.astype(np.uint8)

    if processed_img.dtype != np.uint8:
        if processed_img.max() <= 1.0:
            processed_img = (processed_img * 255).astype(np.uint8)
        else:
            processed_img = processed_img.astype(np.uint8)

    ax_orig.imshow(original_img)
    ax_orig.set_title("Original Input")
    ax_orig.axis("off")

    display_proc = processed_img.copy()
    if vessel_mask is not None:
        if vessel_mask.shape[:2] != processed_img.shape[:2]:
            vessel_mask = cv2.resize(vessel_mask, (processed_img.shape[1], processed_img.shape[0]))
        mask_norm = vessel_mask.astype(float) / 255.0 if vessel_mask.max() > 1 else vessel_mask.astype(float)
        overlay = display_proc.copy()
        overlay[mask_norm > 0.5] = [75, 0, 130]  # Deep Purple (#4B0082)
        cv2.addWeighted(overlay, 0.5, display_proc, 0.5, 0, display_proc)

    ax_proc.imshow(display_proc)

    proc_title = "Processed Input + Vessels"
    if quality_score:
        # Assuming quality_score is a dict or value
        if isinstance(quality_score, dict):
            qs = quality_score.get("vessel_density", 0)
        else:
            qs = quality_score
        proc_title += f"\nVessel Density: {qs:.4f}"

    ax_proc.set_title(proc_title)
    ax_proc.axis("off")

    heatmap_resized = cv2.resize(heatmap, (processed_img.shape[1], processed_img.shape[0]))
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)

    processed_gray = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
    ax_cam.imshow(processed_gray, cmap="gray")
    im = ax_cam.imshow(heatmap_norm, cmap="plasma", alpha=0.6)

    pred_idx = -1
    if class_names and class_name in class_names:
        pred_idx = class_names.index(class_name)

    title_color = "black"
    if 0 <= pred_idx < len(THEME_COLORS):
        title_color = THEME_COLORS[pred_idx]

    title = f"Pred: {class_name}\nConf: {confidence:.1%}"
    if true_label:
        status = "Correct" if true_label == class_name else "Incorrect"
        title += f" ({status})"

    t = ax_cam.set_title(title, fontsize=14, fontweight="bold")
    t.set_color(title_color)

    ax_cam.axis("off")

    ax_cbar = fig.add_subplot(gs[0, 3])
    plt.colorbar(im, cax=ax_cbar)

    if probs is not None and class_names is not None:
        bars = ax_probs.bar(class_names, probs, color=THEME_COLORS, alpha=0.8)
        ax_probs.set_ylim(0, 1.1)
        ax_probs.set_title("Prediction Confidence by Class")
        ax_probs.set_ylabel("Probability")
        ax_probs.grid(axis="y", alpha=0.3)
        ax_probs.spines["top"].set_visible(False)
        ax_probs.spines["right"].set_visible(False)

        for bar in bars:
            height = bar.get_height()
            ax_probs.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.1%}",
                ha="center",
                va="bottom",
                fontweight="bold",
            )
    else:
        ax_probs.axis("off")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    _save_plot(fig, f"Inference_{class_name}_{confidence:.2f}_{np.random.randint(0, 1000)}", save_folder)
    return fig


def plot_binary_classification_metrics(
    results_df: pd.DataFrame, save_folder: str = "evaluation", title_prefix: str = ""
):
    """Compute and plot binary classification metrics (Healthy vs. Diseased).

    Diseased = Class 1-4 (Mild+), Healthy = Class 0 (No DR).

    Args:
        results_df: DataFrame with ``true``, ``pred``, and optional ``prob_*`` columns.
        save_folder: Sub-folder under ``reports/figures/``.
        title_prefix: Prefix prepended to plot titles.

    Returns:
        Matplotlib figure.
    """
    logger.info("--- Binary Classification Analysis (Healthy vs. Diseased) ---")

    y_true_bin = (results_df["true"] > 0).astype(int)
    y_pred_bin = (results_df["pred"] > 0).astype(int)

    prob_cols = [c for c in results_df.columns if c.startswith("prob_")]
    y_prob_bin = None
    if len(prob_cols) == 5 and "prob_0" in results_df.columns:
        y_prob_bin = 1.0 - results_df["prob_0"]

    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)  # Sensitivity
    spec = recall_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0)  # Specificity
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    logger.info(f"Binary Accuracy: {acc:.4f}")
    logger.info(f"Sensitivity (Recall): {rec:.4f} (Ability to detect disease)")
    logger.info(f"Specificity: {spec:.4f} (Ability to confirm health)")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"F1-Score: {f1:.4f}")

    if y_prob_bin is not None:
        try:
            auc_score = roc_auc_score(y_true_bin, y_prob_bin)
            logger.info(f"Binary ROC-AUC: {auc_score:.4f}")
        except ValueError:
            logger.warning("Binary ROC-AUC: Undefined (only one class present in y_true)")

    cm_bin = confusion_matrix(y_true_bin, y_pred_bin)
    cmap_purple = LinearSegmentedColormap.from_list("custom_purple", ["#fafafa", "#4B0082"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 1. Confusion Matrix
    sns.heatmap(
        cm_bin,
        annot=True,
        fmt="d",
        cmap=cmap_purple,
        xticklabels=["Healthy", "Diseased"],
        yticklabels=["Healthy", "Diseased"],
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    axes[0].set_title(f"{title_prefix}Referral Confusion Matrix")

    # 2. ROC Curve
    if y_prob_bin is not None:
        try:
            fpr, tpr, _ = roc_curve(y_true_bin, y_prob_bin)
            roc_auc = auc(fpr, tpr)

            axes[1].plot(fpr, tpr, color="#FF8C00", lw=2, label=f"ROC curve (area = {roc_auc:.2f})")
            axes[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
            axes[1].set_xlim([0.0, 1.0])
            axes[1].set_ylim([0.0, 1.05])
            axes[1].set_xlabel("False Positive Rate")
            axes[1].set_ylabel("True Positive Rate")
            axes[1].set_title("Receiver Operating Characteristic (Binary)")
            axes[1].legend(loc="lower right")
            axes[1].grid(alpha=0.3)
        except ValueError:
            axes[1].text(0.5, 0.5, "ROC Undefined (Single Class)", ha="center")
    else:
        axes[1].text(0.5, 0.5, "Probabilities not available", ha="center")
        axes[1].axis("off")

    plt.tight_layout()
    _save_plot(fig, f"{title_prefix}binary_performance_panel", save_folder)
    return fig


def plot_calibration_curves(y_true, prob_matrix, n_classes=5, save_folder="evaluation", title_prefix="", n_bins=10):
    """Plot calibration curves (reliability diagrams) for each class.

    Args:
        y_true: Array of true class labels.
        prob_matrix: (N, n_classes) probability matrix.
        n_classes: Number of classes.
        save_folder: Sub-folder under ``reports/figures/``.
        title_prefix: Prefix prepended to plot titles.
        n_bins: Number of bins for calibration curve.

    Returns:
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    n_classes = prob_matrix.shape[1]
    for i in range(n_classes):
        y_true_binary = (y_true == i).astype(int)
        prob_class = prob_matrix[:, i]

        fraction_of_positives, mean_predicted_value = calibration_curve(y_true_binary, prob_class, n_bins=n_bins)

        ax.plot(
            mean_predicted_value,
            fraction_of_positives,
            "s-",
            alpha=1.0,
            linewidth=2,
            label=f"Class {i}",
            color=THEME_COLORS[i % len(THEME_COLORS)],
        )

    # Aggregate (lumped) calibration across all classes
    y_true_onehot = np.zeros_like(prob_matrix)
    y_true_onehot[np.arange(len(y_true)), y_true] = 1

    all_outcomes = y_true_onehot.flatten()
    all_probs = prob_matrix.flatten()

    prob_true_curve, prob_pred_curve = calibration_curve(all_outcomes, all_probs, n_bins=n_bins, strategy="uniform")

    # Manual binning to compute SE per bin
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.clip(np.digitize(all_probs, bins) - 1, 0, n_bins - 1)
    bin_total = np.bincount(binids, minlength=n_bins)
    mask = bin_total > 0

    if np.sum(mask) == len(prob_true_curve):
        n = bin_total[mask]
        se = np.sqrt(prob_true_curve * (1 - prob_true_curve) / n)
        ax.plot(prob_pred_curve, prob_true_curve, "o-", color="black", linewidth=3, label="All Classes", zorder=10)
        ax.fill_between(
            prob_pred_curve, prob_true_curve - se, prob_true_curve + se, color="black", alpha=0.15, zorder=9
        )
    else:
        ax.plot(prob_pred_curve, prob_true_curve, "o-", color="black", linewidth=2.5, label="All Classes")

    # Ideal Line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.8, label="Perfectly Calibrated")

    ax.set_ylabel("Fraction of Positives")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_title(f"{title_prefix}Calibration Curves (Reliability Diagram)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    _save_plot(fig, f"{title_prefix}calibration_curves", save_folder)
    return fig


def plot_error_distribution(y_true, y_pred, confidences=None, save_folder="evaluation", title_prefix=""):
    """Analyze the magnitude of prediction errors.

    Args:
        y_true: Array of true labels.
        y_pred: Array of predicted labels.
        confidences: Optional (N,) array of max-probability confidences.
        save_folder: Sub-folder under ``reports/figures/``.
        title_prefix: Prefix prepended to plot titles.

    Returns:
        Matplotlib figure.
    """
    errors = y_pred - y_true
    unique, counts = np.unique(errors, return_counts=True)

    # Create DataFrame for analysis
    data = {"Error Magnitude": errors, "Validation": np.ones_like(errors)}
    if confidences is not None:
        data["Confidence"] = confidences

    df_analysis = pd.DataFrame(data)
    df_analysis["Type"] = df_analysis["Error Magnitude"].apply(
        lambda x: "Correct" if x == 0 else ("Off-by-1" if abs(x) == 1 else "Off-by-many")
    )

    # Color mapping
    palette = {"Correct": "#4B0082", "Off-by-1": "#CC82C4", "Off-by-many": "#FF8C00"}

    # Setup Figure
    if confidences is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    else:
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = None

    err_counts = df_analysis["Error Magnitude"].value_counts().reset_index()
    err_counts.columns = ["Error Magnitude", "Count"]
    # Add Type column back
    err_counts["Type"] = err_counts["Error Magnitude"].apply(
        lambda x: "Correct" if x == 0 else ("Off-by-1" if abs(x) == 1 else "Off-by-many")
    )

    sns.barplot(data=err_counts, x="Error Magnitude", y="Count", hue="Type", palette=palette, dodge=False, ax=ax1)

    # Calculate stats
    total = len(y_true)
    perfect = np.sum(errors == 0)
    off_by_one = np.sum(np.abs(errors) == 1)
    severe = np.sum(np.abs(errors) > 1)

    stats_text = (
        f"Total: {total}\n"
        f"Exact Accuracy: {perfect / total:.1%}\n"
        f"Off-by-1: {off_by_one / total:.1%}\n"
        f"Off-by-many: {severe / total:.1%}"
    )

    ax1.text(
        0.02,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        va="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
    )

    ax1.set_title(f"{title_prefix}Error Magnitude Distribution")
    ax1.grid(axis="y", alpha=0.3)

    if ax2 is not None:
        sns.barplot(
            data=df_analysis,
            x="Error Magnitude",
            y="Confidence",
            hue="Type",
            palette=palette,
            errorbar="se",  # Shows Standard Error
            capsize=0.1,
            dodge=False,
            ax=ax2,
        )
        ax2.set_title(f"{title_prefix}Confidence vs. Error Magnitude")
        ax2.set_ylabel("Mean Confidence Score")
        ax2.grid(axis="y", alpha=0.3)
        if ax2.legend_:
            ax2.legend_.remove()

    plt.tight_layout()
    _save_plot(fig, f"{title_prefix}error_distribution", save_folder)
    return fig


def plot_quality_score_distribution(
    df: pd.DataFrame,
    score_col: str = "quality_score",
    quantile: float = 0.05,
    save_folder: str = "qc",
) -> float:
    """Plot a histogram of image quality scores with a rejection threshold.

    Args:
        df: DataFrame containing the quality-score column.
        score_col: Name of the quality-score column.
        quantile: Percentile for the rejection threshold (e.g. 0.05 = 5th).
        save_folder: Sub-folder under ``reports/figures/``.

    Returns:
        Computed threshold value.
    """
    if score_col not in df.columns:
        logger.warning(f"Column '{score_col}' not found.")
        return 0.0

    set_plot_style()
    threshold = df[score_col].quantile(quantile)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(df[score_col], kde=True, bins=50, color="#4B0082", ax=ax)
    ax.axvline(
        threshold,
        color="#FF8C00",
        linestyle="--",
        label=f"{quantile:.0%} Percentile ({threshold:.4f})",
    )
    ax.set_title("Distribution of Image Quality Scores (Vessel Density)")
    ax.set_xlabel("Vessel Density Score")
    ax.set_ylabel("Count")
    ax.legend()

    _save_plot(fig, "quality_score_distribution", save_folder)
    logger.info(f"Suggested Threshold (exclude below): {threshold:.4f}")
    return threshold


def plot_confusion_matrix_pair(
    y_true,
    y_pred,
    class_names: Optional[List[str]] = None,
    title_prefix: str = "",
    save_folder: str = "evaluation",
):
    """Plot count and row-normalised confusion matrices side by side.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        class_names: Display names for each class.
        title_prefix: Prefix prepended to saved filename.
        save_folder: Sub-folder under ``reports/figures/``.

    Returns:
        Matplotlib figure.
    """
    if class_names is None:
        class_names = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]

    set_plot_style()

    cm = confusion_matrix(y_true, y_pred)
    acc = accuracy_score(y_true, y_pred)

    class_counts = cm.sum(axis=1)
    yticklabels = [f"{n}\n(n={c})" for n, c in zip(class_names, class_counts, strict=True)]

    cmap_purple = LinearSegmentedColormap.from_list("custom_purple", ["#fafafa", "#4B0082"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Count matrix
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap_purple,
        xticklabels=class_names,
        yticklabels=yticklabels,
        annot_kws={"weight": "bold"},
        ax=ax1,
    )
    ax1.set_xlabel("Predicted Label", fontsize=12)
    ax1.set_ylabel("True Label", fontsize=12)
    ax1.set_title(f"Confusion Matrix (Acc: {acc:.1%})", fontsize=14)

    # Normalised matrix
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap=cmap_purple,
        xticklabels=class_names,
        yticklabels=yticklabels,
        annot_kws={"weight": "bold"},
        ax=ax2,
    )
    ax2.set_xlabel("Predicted Label", fontsize=12)
    ax2.set_ylabel("True Label", fontsize=12)
    ax2.set_title("Normalized Confusion Matrix (Recall)", fontsize=14)

    plt.tight_layout()
    _save_plot(fig, f"{title_prefix}confusion_matrices", save_folder)
    return fig


def plot_performance_vs_quality(
    results_df,
    manifest_df,
    quality_col="quality_score",
    save_folder="evaluation",
    title_prefix="",
    n_bins=10,
    min_samples=50,
):
    """Plot model accuracy vs. image quality with error bars.

    Args:
        results_df: Predictions DataFrame (must contain ``image_name``, ``true``, ``pred``).
        manifest_df: Manifest DataFrame with a quality score column.
        quality_col: Name of the quality column in *manifest_df*.
        save_folder: Sub-folder under ``reports/figures/``.
        title_prefix: Prefix prepended to plot titles.
        n_bins: Number of equal-width bins.
        min_samples: Minimum samples per bin to include.

    Returns:
        Matplotlib figure or ``None`` if data is insufficient.
    """
    if "image_name" not in results_df.columns:
        logger.error("results_df missing 'image_name' column")
        return None

    left_key = "image_name"
    right_key = "image" if "image" in manifest_df.columns else "image_name"

    if right_key not in manifest_df.columns:
        logger.error(f"manifest_df missing key column {right_key}")
        return None

    merged = pd.merge(results_df, manifest_df, left_on=left_key, right_on=right_key, how="inner")

    if merged.empty:
        logger.warning("Merge result is empty. Check filenames matching.")
        return None

    if quality_col not in merged.columns:
        logger.warning(f"Quality column '{quality_col}' not found in merged data.")
        return None

    merged["is_correct"] = (merged["true"] == merged["pred"]).astype(int)
    merged["quality_bin"] = pd.cut(merged[quality_col], bins=n_bins)

    bin_stats = (
        merged.groupby("quality_bin", observed=False)
        .agg({quality_col: "mean", "is_correct": ["mean", "count", "sem"]})
        .reset_index()
    )

    bin_stats.columns = ["bin", "mean_qual", "accuracy", "count", "se"]
    valid_count = len(bin_stats[bin_stats["count"] >= min_samples])
    logger.info(f"Total bins: {len(bin_stats)}. Valid bins (>= {min_samples} samples): {valid_count}")
    bin_stats = bin_stats[bin_stats["count"] >= min_samples]

    if bin_stats.empty:
        logger.warning(f"No bins have >= {min_samples} samples. Cannot plot performance vs quality.")
        return None

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    bin_width = (merged[quality_col].max() - merged[quality_col].min()) / n_bins
    ax2.bar(
        bin_stats["mean_qual"],
        bin_stats["count"],
        width=bin_width * 0.8,
        alpha=1.0,
        color="#FF8C00",
        label="Sample Count",
    )

    ax1.errorbar(
        bin_stats["mean_qual"],
        bin_stats["accuracy"],
        yerr=bin_stats["se"],
        fmt="-o",
        color="#4B0082",
        ecolor="#4B0082",
        elinewidth=2,
        capsize=4,
        linewidth=2,
        label="Accuracy ± SE",
    )

    # Draw accuracy line on top of the bars
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)
    ax1.set_xlabel(f"Image Quality ({quality_col})")
    ax1.set_ylabel("Model Accuracy")
    ax2.set_ylabel("Number of Samples")
    ax1.set_title(f"{title_prefix}Performance vs. Image Quality (Equal Bins)")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0.0, 1.05)
    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, labels + bar_labels, loc="lower right")

    _save_plot(fig, f"{title_prefix}perf_vs_{quality_col}", save_folder)
    return fig
