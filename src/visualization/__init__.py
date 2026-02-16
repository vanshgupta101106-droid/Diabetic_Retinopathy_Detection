"""
Visualization Module
====================

Exports the key plotting functions for notebooks and scripts.
"""

from src.visualization.plots import (
    THEME_COLORS,
    plot_binary_classification_metrics,
    plot_calibration_curves,
    plot_class_distribution,
    plot_confidence_abstention_panel,
    plot_confusion_matrix_pair,
    plot_error_distribution,
    plot_image_grid,
    plot_inference_result,
    plot_learning_curves,
    plot_model_diagnostics,
    plot_performance_vs_quality,
    plot_quality_score_distribution,
    set_plot_style,
)

from src.visualization.eda import (
    create_vessel_overlay,
    load_and_show_error_samples,
    load_and_show_preprocessing,
    load_and_show_quality_examples,
    print_classification_summary,
    show_augmentation_verification,
    show_vessel_density_report,
    test_api_and_visualize,
)
