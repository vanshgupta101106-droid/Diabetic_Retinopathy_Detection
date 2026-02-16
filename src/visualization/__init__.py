"""
Visualization Module
====================

Exports the key plotting functions for notebooks and scripts.
"""

from src.visualization.eda import (
    create_vessel_overlay as create_vessel_overlay,
)
from src.visualization.eda import (
    load_and_show_error_samples as load_and_show_error_samples,
)
from src.visualization.eda import (
    load_and_show_preprocessing as load_and_show_preprocessing,
)
from src.visualization.eda import (
    load_and_show_quality_examples as load_and_show_quality_examples,
)
from src.visualization.eda import (
    print_classification_summary as print_classification_summary,
)
from src.visualization.eda import (
    show_augmentation_verification as show_augmentation_verification,
)
from src.visualization.eda import (
    show_vessel_density_report as show_vessel_density_report,
)
from src.visualization.eda import (
    test_api_and_visualize as test_api_and_visualize,
)
from src.visualization.plots import (
    THEME_COLORS as THEME_COLORS,
)
from src.visualization.plots import (
    plot_binary_classification_metrics as plot_binary_classification_metrics,
)
from src.visualization.plots import (
    plot_calibration_curves as plot_calibration_curves,
)
from src.visualization.plots import (
    plot_class_distribution as plot_class_distribution,
)
from src.visualization.plots import (
    plot_confidence_abstention_panel as plot_confidence_abstention_panel,
)
from src.visualization.plots import (
    plot_confusion_matrix_pair as plot_confusion_matrix_pair,
)
from src.visualization.plots import (
    plot_error_distribution as plot_error_distribution,
)
from src.visualization.plots import (
    plot_image_grid as plot_image_grid,
)
from src.visualization.plots import (
    plot_inference_result as plot_inference_result,
)
from src.visualization.plots import (
    plot_learning_curves as plot_learning_curves,
)
from src.visualization.plots import (
    plot_model_diagnostics as plot_model_diagnostics,
)
from src.visualization.plots import (
    plot_performance_vs_quality as plot_performance_vs_quality,
)
from src.visualization.plots import (
    plot_quality_score_distribution as plot_quality_score_distribution,
)
from src.visualization.plots import (
    set_plot_style as set_plot_style,
)
