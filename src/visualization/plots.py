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
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from cycler import cycler
import numpy as np
import pandas as pd

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
    clean_title = title.lower().replace(" ", "_").replace(":", "").replace("-", "_")
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
    # Reduce whitespace: closer title (rect top 0.97) and tighter row spacing (h_pad 1.5)
    plt.tight_layout(rect=[0, 0, 1, 0.97], h_pad=1.5, w_pad=0.5)
    _save_plot(fig, title, save_folder)
    return fig
