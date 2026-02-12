"""
Diabetic Retinopathy Detection - Visualization Module
=====================================================

This module provides a unified styling and plotting interface for the project.
It adheres to a strict "Deep Purple & Orange" color theme for consistent,
publication-ready figures.

Functions:
    set_plot_style: Configures global matplotlib/seaborn defaults.
    plot_image_grid: Visualizes a grid of retinal fundus images.
    plot_class_distribution: Plots the distribution of DR severities.
    plot_learning_curves: Plots training history.
    plot_model_diagnostics: Comprehensive evaluation panel.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import warnings
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
import pandas as pd
from cycler import cycler
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats

from src.config import REPORTS_DIR

logger = logging.getLogger(__name__)

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
    """
    Sets the project-wide matplotlib and seaborn style.

    Theme:
        Primary: Deep Purple (#4B0082)
        Contrast: Dark Orange (#FF8C00)
    """
    plt.style.use("ggplot")

    # Custom Palette defined in design specs
    # Ordered from Deep Purple (Healthiest) to Dark Orange (Severe)
    selected_colors = [
        "#4B0082",  # 0: Deep Purple
        "#762B98",  # 1: Purple
        "#A157AE",  # 2: Medium Orchid
        "#CC82C4",  # 3: Pinkish
        "#FF8C00",  # 4: Dark Orange
    ]

    # Configure Cycle
    color_cycle = cycler("color", selected_colors)
    mpl.rcParams["axes.prop_cycle"] = color_cycle

    # Configure Seaborn
    sns.set_palette(sns.color_palette(selected_colors))

    # Typography & Lines
    mpl.rcParams["font.family"] = "sans-serif"
    mpl.rcParams["axes.titlesize"] = 14
    mpl.rcParams["axes.labelsize"] = 12
    mpl.rcParams["lines.linewidth"] = 2.0
    mpl.rcParams["grid.alpha"] = 0.5
    
    # Remove top and right spines
    mpl.rcParams["axes.spines.top"] = False
    mpl.rcParams["axes.spines.right"] = False

    return selected_colors

def plot_class_distribution(
    df: pd.DataFrame, 
    label_col: str = "level", 
    title: str = "Class Epidemiology",
    save_folder: str = "eda"
):
    """
    Plots the count of each DR severity level.
    """
    colors = set_plot_style()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Count plot
    sns.countplot(x=label_col, data=df, palette=colors, ax=ax)
    
    # Add percentages
    total = len(df)
    for p in ax.patches:
        percentage = f'{100 * p.get_height() / total:.1f}%'
        x = p.get_x() + p.get_width() / 2
        y = p.get_height()
        ax.annotate(percentage, (x, y), ha='center', va='bottom')
    
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
    save_folder: str = "eda"
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
            ax.imshow(images[i], aspect='equal')
            
            # Determine class color
            label_val = labels[i]
            try:
                c_idx = int(label_val)
                c_idx = max(0, min(c_idx, 4))
                class_color = colors[c_idx]
            except:
                class_color = 'black'

            if titles:
                ax_title = titles[i]
            else:
                ax_title = f"Grade: {labels[i]}"
                if preds is not None:
                    ax_title += f"\npred: {preds[i]}"
            
            ax.set_title(ax_title, fontsize=12, color=class_color, fontweight='bold')
            
            # Add colored border (Frame)
            # Turn spines on explicitly for the frame effect
            for side in ['top', 'bottom', 'left', 'right']:
                ax.spines[side].set_visible(True)
                ax.spines[side].set_color(class_color)
                ax.spines[side].set_linewidth(4)
            
            # Remove ticks but keep the colored frame
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False) # Disable grid inside image
            
        else:
            ax.axis('off')
            
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
    n_plots = len(metrics)
    
    if n_plots == 0:
        return
        
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 5))
    if n_plots == 1:
        axes = [axes]
        
    epochs = range(1, len(history[metrics[0]]) + 1)
    
    # Colors
    col_train = "#4B0082"  # Deep Purple
    col_val = "#FF8C00"    # Dark Orange
    
    for ax, metric in zip(axes, metrics):
        metric_name = metric.replace("train_", "")
        val_metric = f"val_{metric_name}"
        
        ax.plot(epochs, history[metric], label=f"Train {metric_name}", color=col_train, linewidth=2)
        if val_metric in history:
            ax.plot(epochs, history[val_metric], label=f"Val {metric_name}", color=col_val, linewidth=2, linestyle="--")
            
        ax.set_title(f"{metric_name.capitalize()} Evolution")
        ax.set_xlabel("Epochs")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
    plt.tight_layout()
    _save_plot(fig, f"{title_prefix}Learning_Curves", save_folder)
    return fig

def _add_heatmap_annotations(ax, data, fmt="d", threshold=0.3):
    """
    Manually adds text to heatmap cells.
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
            
            # Determine text color based on background intensity
            ratio = val / max_val
            color = "white" if ratio > threshold else "black"
            
            ax.text(j + 0.5, i + 0.5, text_val, 
                   ha="center", va="center", color=color, fontweight="bold")

def plot_model_diagnostics(
    results_df: pd.DataFrame, 
    n_classes: int = 5,
    save_folder: str = "evaluation", 
    title_prefix: str = ""
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
    
    # Color map for Heatmap
    cmap_purple = LinearSegmentedColormap.from_list("custom_purple", ["#fafafa", "#4B0082"])

    # --- 1. Confusion Matrix ---
    ax_cm = fig.add_subplot(gs[0])
    cm = confusion_matrix(results_df["true"], results_df["pred"], labels=list(range(n_classes)))
    
    # Normalize for coloring, but Annotate counts
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm) # Handle division by zero
    
    sns.heatmap(cm_norm, annot=False, cmap=cmap_purple, ax=ax_cm, cbar=False,
               xticklabels=[str(i) for i in range(n_classes)],
               yticklabels=[str(i) for i in range(n_classes)])
    
    _add_heatmap_annotations(ax_cm, cm, fmt="d", threshold=0.3)
    
    acc = accuracy_score(results_df["true"], results_df["pred"])
    ax_cm.set_title(f"Confusion Matrix (Acc: {acc:.1%})")
    ax_cm.set_ylabel("True Label")
    ax_cm.set_xlabel("Predicted Label")

    # --- 2. ROC Curves (One-vs-Rest) ---
    ax_roc = fig.add_subplot(gs[1])
    
    # We need probabilities for each class. 
    # results_df 'prob' might be a list or we need separate columns
    # Assuming results_df has a col 'probs' which is a list/array
    
    # If prob is a string representation of list, parsing might be needed.
    # Checks if 'prob_0', 'prob_1' etc exist
    prob_cols = [c for c in results_df.columns if c.startswith("prob_")]
    if len(prob_cols) == n_classes:
        y_score = results_df[prob_cols].values
        
        # Binarize labels
        from sklearn.preprocessing import label_binarize
        y_test = label_binarize(results_df["true"], classes=range(n_classes))
        
        for i in range(n_classes):
             fpr, tpr, _ = roc_curve(y_test[:, i], y_score[:, i])
             roc_auc = auc(fpr, tpr)
             ax_roc.plot(fpr, tpr, label=f'Class {i} (AUC = {roc_auc:.2f})', lw=2)
             
        ax_roc.plot([0, 1], [0, 1], 'k--', lw=1)
        ax_roc.set_title("ROC Curves (One-vs-Rest)")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend(loc="lower right")
    else:
        ax_roc.text(0.5, 0.5, "Probability data missing\nfor ROC", ha='center')

    # --- 3. Class-wise Performance (e.g. Sensitivity/Specificity/F1) ---
    ax_bar = fig.add_subplot(gs[2])
    
    # Calculate per-class accuracy (Recall)
    cm_diag = cm.diagonal()
    cm_sum = cm.sum(axis=1)
    recall = np.divide(cm_diag, cm_sum, out=np.zeros_like(cm_diag, dtype=float), where=cm_sum!=0)
    
    sns.barplot(x=np.arange(n_classes), y=recall, ax=ax_bar, palette=colors)
    ax_bar.set_ylim(0, 1)
    ax_bar.set_xlabel("Class")
    ax_bar.set_ylabel("Recall (Sensitivity)")
    ax_bar.set_title("Per-Class Recall")
    
    for i, v in enumerate(recall):
        ax_bar.text(i, v + 0.02, f"{v:.2f}", ha='center')
        
    _save_plot(fig, f"{title_prefix}Model_Diagnostics", save_folder)
    return fig

def plot_confidence_abstention_panel(
    results_df: pd.DataFrame,
    confidence_threshold: float = 0.7,
    save_folder: str = "evaluation",
    title_prefix: str = ""
):
    """
    Visualizes confidence distribution and impact of abstention.
    Adapts logic for Multi-class: Confidence is max(probability vector).
    """
    set_plot_style()
    
    # Extract max confidence
    prob_cols = [c for c in results_df.columns if c.startswith("prob_")]
    if not prob_cols:
        return
        
    probs = results_df[prob_cols].values
    confidence = np.max(probs, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Confidence Histogram
    ax_hist = axes[0]
    sns.histplot(confidence, bins=20, ax=ax_hist, kde=True, color="#4B0082")
    ax_hist.axvline(confidence_threshold, color="#FF8C00", linestyle="--", label=f"Threshold {confidence_threshold}")
    ax_hist.set_title("Prediction Confidence Distribution")
    ax_hist.set_xlabel("Max Probability (Confidence)")
    ax_hist.legend()
    
    # 2. Accuracy vs Retention Curve
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
