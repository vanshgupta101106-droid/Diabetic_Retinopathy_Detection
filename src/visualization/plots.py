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
import cv2  # Added cv2 import
from cycler import cycler
from sklearn.metrics import (
    confusion_matrix, 
    roc_curve, 
    auc, 
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score, 
    roc_auc_score,
    cohen_kappa_score
)
from sklearn.calibration import calibration_curve
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
    col_val = "#FF8C00"    # Dark Orange
    
    for ax, metric in zip(axes, all_metrics_to_plot):
        if metric.startswith("train_"):
            metric_name = metric.replace("train_", "")
            val_metric = f"val_{metric_name}"
            
            ax.plot(epochs, history[metric], label=f"Train {metric_name}", color=col_train, linewidth=2)
            if val_metric in history:
                ax.plot(epochs, history[val_metric], label=f"Val {metric_name}", color=col_val, linewidth=2, linestyle="--")
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
    patient_info: Optional[Dict[str, Any]] = None
):
    """
    Visualizes inference result: Original, Processed, and Grad-CAM (Plasma).
    """
    # Create Figure with GridSpec
    fig = plt.figure(figsize=(20, 10))
    gs = fig.add_gridspec(2, 4, width_ratios=[1, 1, 1, 0.05])
    
    # Suptitle with Patient Info
    if patient_info:
        info_str = " | ".join([f"{k}: {v}" for k, v in patient_info.items()])
        fig.suptitle(info_str, fontsize=16, y=0.98, fontweight='bold')
    
    ax_orig = fig.add_subplot(gs[0, 0])
    ax_proc = fig.add_subplot(gs[0, 1])
    ax_cam = fig.add_subplot(gs[0, 2])
    # cbar ax is gs[0, 3] handled by colorbar
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

    # 1. Original
    ax_orig.imshow(original_img)
    ax_orig.set_title("Original Input")
    ax_orig.axis('off')
    
    # 2. Processed with Vessel Overlay
    # Create an RGB version of the vessel mask (green)
    display_proc = processed_img.copy()
    
    if vessel_mask is not None:
        # Resize mask if needed
        if vessel_mask.shape[:2] != processed_img.shape[:2]:
            vessel_mask = cv2.resize(vessel_mask, (processed_img.shape[1], processed_img.shape[0]))
            
        # Create a purple overlay
        # Normalize mask to 0-1
        mask_norm = vessel_mask.astype(float) / 255.0 if vessel_mask.max() > 1 else vessel_mask.astype(float)
        
        # Overlay: Tint Deep Purple where vessels are present
        overlay = display_proc.copy()
        # Using Deep Purple: (75, 0, 130) but let's make it brighter for overlay: (147, 112, 219) MediumPurple 
        # or pure magenta/purple (255, 0, 255) for contrast? 
        # User asked for "same purple color as we did in notebook 1", which refers to the theme #4B0082.
        # RGB: (75, 0, 130)
        overlay[mask_norm > 0.5] = [75, 0, 130] 
        
        # Blend
        cv2.addWeighted(overlay, 0.5, display_proc, 0.5, 0, display_proc)
        
    ax_proc.imshow(display_proc)
    
    proc_title = "Processed Input + Vessels"
    if quality_score:
        # Assuming quality_score is a dict or value
        if isinstance(quality_score, dict):
            qs = quality_score.get('vessel_density', 0)
        else:
            qs = quality_score
        proc_title += f"\nVessel Density: {qs:.4f}"
        
    ax_proc.set_title(proc_title)
    ax_proc.axis('off')
    
    # 3. Grad-CAM with Plasma
    # Resize heatmap
    heatmap_resized = cv2.resize(heatmap, (processed_img.shape[1], processed_img.shape[0]))
    
    # Normalize Heatmap (0-1)
    heatmap_norm = (heatmap_resized - heatmap_resized.min()) / (heatmap_resized.max() - heatmap_resized.min() + 1e-8)
    
    # Background
    processed_gray = cv2.cvtColor(processed_img, cv2.COLOR_RGB2GRAY)
    ax_cam.imshow(processed_gray, cmap='gray') 
    
    # Overlay Heatmap
    im = ax_cam.imshow(heatmap_norm, cmap='plasma', alpha=0.6)
    
    # Title Color Logic
    theme_colors = ['#4B0082', '#762B98', '#A157AE', '#CC82C4', '#FF8C00']
    
    # Find index of predicted class
    pred_idx = -1
    if class_names and class_name in class_names:
        pred_idx = class_names.index(class_name)
    
    title_color = 'black'
    if pred_idx >= 0 and pred_idx < len(theme_colors):
        title_color = theme_colors[pred_idx]
        
    title = f"Pred: {class_name}\nConf: {confidence:.1%}"
    
    if true_label:
        status = "Correct" if true_label == class_name else "Incorrect"
        title += f" ({status})"
            
    # Set title with the predicted class color
    t = ax_cam.set_title(title, fontsize=14, fontweight='bold')
    t.set_color(title_color)
    
    ax_cam.axis('off')
    
    # Colorbar
    ax_cbar = fig.add_subplot(gs[0, 3])
    plt.colorbar(im, cax=ax_cbar)
    
    # 4. Probabilities
    if probs is not None and class_names is not None:
        bars = ax_probs.bar(class_names, probs, color=theme_colors, alpha=0.8)
        ax_probs.set_ylim(0, 1.1)
        ax_probs.set_title("Prediction Confidence by Class")
        ax_probs.set_ylabel("Probability")
        ax_probs.grid(axis='y', alpha=0.3)
        ax_probs.spines['top'].set_visible(False)
        ax_probs.spines['right'].set_visible(False)
        
        for bar in bars:
            height = bar.get_height()
            ax_probs.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                        f'{height:.1%}', ha='center', va='bottom', fontweight='bold')
    else:
        ax_probs.axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust for suptitle

    # Unique filename based on class and confidence (avoid overwriting)
    _save_plot(fig, f"Inference_{class_name}_{confidence:.2f}_{np.random.randint(0,1000)}", save_folder)
    return fig


def plot_binary_classification_metrics(results_df: pd.DataFrame, 
                                     save_folder: str = "reports/figures", 
                                     title_prefix: str = ""):
    """
    Computes and plots binary classification metrics (Healthy vs. Diseased).
    Diseased = Class 1-4 (Mild+), Healthy = Class 0 (No DR).
    """

    print("--- Binary Classification Analysis (Healthy vs. Diseased) ---")

    # Binarize labels: 0 -> 0 (Healthy), 1-4 -> 1 (Diseased)
    y_true_bin = (results_df['true'] > 0).astype(int)
    y_pred_bin = (results_df['pred'] > 0).astype(int)
    
    # Calculate probability of disease (Sum of probs for classes 1-4)
    # Check if probability columns exist
    prob_cols = [c for c in results_df.columns if c.startswith('prob_')]
    y_prob_bin = None
    if len(prob_cols) == 5:
        # P(Diseaset) = 1 - P(No DR) or Sum(P(Mild)...P(Prolif))
        # Using 1 - P(0) is safer if probs sum to 1
        if 'prob_0' in results_df.columns:
             y_prob_bin = 1.0 - results_df['prob_0']
        else:
             y_prob_bin = results_df[[c for c in prob_cols if c != 'prob_0']].sum(axis=1)

    # Metrics
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0) # Sensitivity
    spec = recall_score(y_true_bin, y_pred_bin, pos_label=0, zero_division=0) # Specificity
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    
    print(f"Binary Accuracy: {acc:.4f}")
    print(f"Sensitivity (Recall): {rec:.4f} (Ability to detect disease)")
    print(f"Specificity: {spec:.4f} (Ability to confirm health)")
    print(f"Precision: {prec:.4f}")
    print(f"F1-Score: {f1:.4f}")

    if y_prob_bin is not None:
        try:
            auc_score = roc_auc_score(y_true_bin, y_prob_bin)
            print(f"Binary ROC-AUC: {auc_score:.4f}")
        except ValueError:
            print("Binary ROC-AUC: Undefined (only one class present in y_true)")

    # Confusion Matrix for Binary Task & ROC Curve
    cm_bin = confusion_matrix(y_true_bin, y_pred_bin)
    
    # Define custom purple colormap to match project theme
    cmap_purple = LinearSegmentedColormap.from_list("custom_purple", ["#fafafa", "#4B0082"])

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 1. Confusion Matrix
    sns.heatmap(cm_bin, annot=True, fmt='d', cmap=cmap_purple,
                xticklabels=['Healthy', 'Diseased'],
                yticklabels=['Healthy', 'Diseased'], ax=axes[0])
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('True')
    axes[0].set_title(f'{title_prefix}Referral Confusion Matrix')

    # 2. ROC Curve
    if y_prob_bin is not None:
        try:
            fpr, tpr, _ = roc_curve(y_true_bin, y_prob_bin)
            roc_auc = auc(fpr, tpr)
            
            axes[1].plot(fpr, tpr, color='#FF8C00', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
            axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            axes[1].set_xlim([0.0, 1.0])
            axes[1].set_ylim([0.0, 1.05])
            axes[1].set_xlabel('False Positive Rate')
            axes[1].set_ylabel('True Positive Rate')
            axes[1].set_title('Receiver Operating Characteristic (Binary)')
            axes[1].legend(loc="lower right")
            axes[1].grid(alpha=0.3)
        except ValueError:
             axes[1].text(0.5, 0.5, "ROC Undefined (Single Class)", ha='center')
    else:
        axes[1].text(0.5, 0.5, "Probabilities not available", ha='center')
        axes[1].axis('off')
    
    plt.tight_layout()
    _save_plot(fig, f"{title_prefix}binary_performance_panel", save_folder)
    return fig


def plot_calibration_curves(y_true, prob_matrix, n_classes=5, save_folder="reports/figures", title_prefix="", n_bins=10):
    """
    Plots calibration curves (reliability diagrams) for each class.
    Checks if predicted probabilities match observed frequencies.
    """
    from sklearn.calibration import calibration_curve # Local import
    fig, ax = plt.subplots(figsize=(10, 8))
    
    colors = ['#4B0082', '#762B98', '#A157AE', '#CC82C4', '#FF8C00']
    
    # 1. Plot Per-Class Calibration
    n_classes = prob_matrix.shape[1]
    for i in range(n_classes):
        # Create binary target for this class (One-vs-Rest)
        y_true_binary = (y_true == i).astype(int)
        prob_class = prob_matrix[:, i]
        
        fraction_of_positives, mean_predicted_value = calibration_curve(y_true_binary, prob_class, n_bins=n_bins)
        
        ax.plot(mean_predicted_value, fraction_of_positives, "s-", alpha=1.0, linewidth=2, 
                label=f'Class {i}', color=colors[i % len(colors)])

    # 2. Plot Aggregate (Lumped) Calibration
    # Flatten all probabilities and outcomes to treat as one large binary problem
    y_true_onehot = np.zeros_like(prob_matrix)
    y_true_onehot[np.arange(len(y_true)), y_true] = 1
    
    all_outcomes = y_true_onehot.flatten()
    all_probs = prob_matrix.flatten()
    
    # Custom binning to get counts for error bars
    # We use calibration_curve to get the points, but we need counts for SE
    # from sklearn.calibration import calibration_curve # Removed redundant import
    prob_true_curve, prob_pred_curve = calibration_curve(all_outcomes, all_probs, n_bins=n_bins, strategy='uniform')

    # To calculate Standard Error, we need sample counts per bin
    # Re-bin manually to match sklearn's strategy='uniform'
    # Sklearn calibration_curve bins [0, 1] into n_bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(all_probs, bins) - 1
    # Fix edge case where prob == 1.0 goes to bin n_bins (index n_bins) -> clip to n_bins-1
    binids = np.clip(binids, 0, n_bins - 1)
    
    bin_total = np.bincount(binids, minlength=n_bins)
    
    # We only care about bins that have data (mask > 0)
    mask = bin_total > 0
    
    # Align our manual binning with sklearn's output
    # prob_true_curve contains values only for populated bins
    if np.sum(mask) == len(prob_true_curve):
         n = bin_total[mask]
         # Standard Error = sqrt(p(1-p)/n)
         se = np.sqrt(prob_true_curve * (1 - prob_true_curve) / n)
         
         # Plot line (solid black)
         ax.plot(prob_pred_curve, prob_true_curve, "o-", color='black', linewidth=3, 
                 label='All Classes', zorder=10)
                 
         # Plot shaded error region
         ax.fill_between(prob_pred_curve, prob_true_curve - se, prob_true_curve + se, 
                         color='black', alpha=0.15, zorder=9) # No label to exclude from legend
    else:
         # Fallback if alignment fails (should be rare/impossible with consistent binning)
         ax.plot(prob_pred_curve, prob_true_curve, "o-", color='black', linewidth=2.5, label='All Classes')

    # Ideal Line
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.8, label="Perfectly Calibrated")

    ax.set_ylabel("Fraction of Positives")
    ax.set_xlabel("Mean Predicted Probability")
    ax.set_title(f"{title_prefix}Calibration Curves (Reliability Diagram)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    _save_plot(fig, f"{title_prefix}calibration_curves", save_folder)
    return fig


def plot_error_distribution(y_true, y_pred, confidences=None, save_folder="reports/figures", title_prefix=""):
    """
    Analyzes the magnitude of errors (e.g., Off-by-1 vs Off-by-3).
    If confidences (array of shape (N,)) are provided, adds a second subplot
    showing the average confidence for each error magnitude.
    """
    errors = y_pred - y_true
    unique, counts = np.unique(errors, return_counts=True)
    
    # Create DataFrame for analysis
    data = {'Error Magnitude': errors, 'Validation': np.ones_like(errors)}
    if confidences is not None:
        data['Confidence'] = confidences
        
    df_analysis = pd.DataFrame(data)
    df_analysis['Type'] = df_analysis['Error Magnitude'].apply(
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

    # --- Plot 1: Count Distribution ---
    # Aggregate counts for plotting
    err_counts = df_analysis['Error Magnitude'].value_counts().reset_index()
    err_counts.columns = ['Error Magnitude', 'Count']
    # Add Type column back
    err_counts['Type'] = err_counts['Error Magnitude'].apply(
        lambda x: "Correct" if x == 0 else ("Off-by-1" if abs(x) == 1 else "Off-by-many")
    )
    
    sns.barplot(data=err_counts, x="Error Magnitude", y="Count", hue="Type", palette=palette, dodge=False, ax=ax1)
    
    # Calculate stats
    total = len(y_true)
    perfect = np.sum(errors == 0)
    off_by_one = np.sum(np.abs(errors) == 1)
    severe = np.sum(np.abs(errors) > 1)
    
    stats_text = (f"Total: {total}\n"
                  f"Exact Accuracy: {perfect/total:.1%}\n"
                  f"Off-by-1: {off_by_one/total:.1%}\n"
                  f"Off-by-many: {severe/total:.1%}")
    
    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes, va='top', 
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    
    ax1.set_title(f"{title_prefix}Error Magnitude Distribution")
    ax1.grid(axis='y', alpha=0.3)
    
    # --- Plot 2: Confidence vs Error Magnitude ---
    if ax2 is not None:
        sns.barplot(
            data=df_analysis, 
            x="Error Magnitude", 
            y="Confidence", 
            hue="Type", 
            palette=palette, 
            errorbar='se',  # Shows Standard Error
            capsize=.1,
            dodge=False, 
            ax=ax2
        )
        ax2.set_title(f"{title_prefix}Confidence vs. Error Magnitude")
        ax2.set_ylabel("Mean Confidence Score")
        ax2.grid(axis='y', alpha=0.3)
        # remove legend from second plot if redundant, or keep it
        ax2.legend_.remove() if ax2.legend_ else None

    plt.tight_layout()
    _save_plot(fig, f"{title_prefix}error_distribution", save_folder)
    return fig


def plot_performance_vs_quality(results_df, manifest_df, quality_col='quality_score', 
                              save_folder="reports/figures", title_prefix="", n_bins=10, min_samples=50):
    """
    Plots model accuracy vs image quality with error bars (Standard Error).
    Uses equally spaced bins for the quality metric.
    """
    # Merge on image name
    if 'image_name' not in results_df.columns:
        logging.error("results_df missing 'image_name' column")
        return None
        
    left_key = 'image_name'
    right_key = 'image' if 'image' in manifest_df.columns else 'image_name'
    
    if right_key not in manifest_df.columns:
         logging.error(f"manifest_df missing key column {right_key}")
         return None

    merged = pd.merge(results_df, manifest_df, left_on=left_key, right_on=right_key, how='inner')
    
    if merged.empty:
        logging.warning("Merge result is empty. Check filenames matching.")
        return None
        
    # Check if quality column exists
    if quality_col not in merged.columns:
        logging.warning(f"Quality column '{quality_col}' not found in merged data.")
        return None

    # Calculate correctness
    merged['is_correct'] = (merged['true'] == merged['pred']).astype(int)

    # Bin the quality metric (Equal Width Bins)
    # We use pd.cut for equal intervals in quality space
    merged['quality_bin'] = pd.cut(merged[quality_col], bins=n_bins)
    
    # Aggregate statistics
    bin_stats = merged.groupby('quality_bin', observed=False).agg({
        quality_col: 'mean',           # Center of the bin (approx)
        'is_correct': ['mean', 'count', 'sem'] # Accuracy, N samples, Standard Error
    }).reset_index()
    
    # Flatten columns
    bin_stats.columns = ['bin', 'mean_qual', 'accuracy', 'count', 'se']
    
    # Remove bins with insufficient samples to avoid noise
    print(f"Total bins: {len(bin_stats)}. Valid bins (>= {min_samples} samples): {len(bin_stats[bin_stats['count'] >= min_samples])}")
    bin_stats = bin_stats[bin_stats['count'] >= min_samples]

    if bin_stats.empty:
        logging.warning(f"No bins have >= {min_samples} samples. Cannot plot performance vs quality.")
        return None

    # Create axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax2 = ax1.twinx()

    # 1. Bar Plot for Sample Count (Secondary Y-Axis) - PLOT FIRST (Background)
    # Calculate width for bars (uniform since we used pd.cut)
    bin_width = (merged[quality_col].max() - merged[quality_col].min()) / n_bins
    
    ax2.bar(
        bin_stats['mean_qual'], 
        bin_stats['count'], 
        width=bin_width * 0.8, 
        alpha=1.0,  # Opaque 
        color='#FF8C00', # Darker Orange
        label='Sample Count'
    )
    
    # 2. Error Bar Plot for Accuracy (Primary Y-Axis) - PLOT SECOND (Foreground)
    ax1.errorbar(
        bin_stats['mean_qual'], 
        bin_stats['accuracy'], 
        yerr=bin_stats['se'], 
        fmt='-o', 
        color='#4B0082', 
        ecolor='#4B0082', 
        elinewidth=2, 
        capsize=4, 
        linewidth=2, 
        label='Accuracy ± SE'
    )
    
    # Ensure ax1 (accuracy) is drawn ON TOP of ax2 (bars)
    ax1.set_zorder(ax2.get_zorder() + 1)
    ax1.patch.set_visible(False)  # Make ax1 background transparent so ax2 shows through
    
    # Labels & Title
    ax1.set_xlabel(f"Image Quality ({quality_col})")
    ax1.set_ylabel("Model Accuracy")
    ax2.set_ylabel("Number of Samples")
    ax1.set_title(f"{title_prefix}Performance vs. Image Quality (Equal Bins)")
    ax1.grid(True, alpha=0.3)
    
    # Axis Limits customization
    ax1.set_ylim(0.0, 1.05)
    
    # Combine legends
    lines, labels = ax1.get_legend_handles_labels()
    bars, bar_labels = ax2.get_legend_handles_labels()
    ax1.legend(lines + bars, labels + bar_labels, loc='lower right')
    
    _save_plot(fig, f"{title_prefix}perf_vs_{quality_col}", save_folder)
    return fig


