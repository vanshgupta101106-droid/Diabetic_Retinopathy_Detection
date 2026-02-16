"""Tests for src.visualization.plots.

Each test verifies a plot function runs without error on synthetic data
and returns a matplotlib Figure.
"""

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for CI

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from src.visualization.plots import (
    plot_class_distribution,
    plot_learning_curves,
    plot_model_diagnostics,
    plot_confidence_abstention_panel,
    plot_binary_classification_metrics,
    plot_calibration_curves,
    plot_error_distribution,
    set_plot_style,
)


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test to avoid memory leaks."""
    yield
    plt.close("all")


@pytest.fixture
def synthetic_results_df():
    """Minimal predictions DataFrame with probability columns."""
    n = 100
    rng = np.random.RandomState(42)
    true = rng.randint(0, 5, n)
    pred = rng.randint(0, 5, n)
    probs = rng.dirichlet(np.ones(5), n)
    df = pd.DataFrame({"true": true, "pred": pred})
    for i in range(5):
        df[f"prob_{i}"] = probs[:, i]
    return df


class TestSetPlotStyle:
    def test_returns_color_list(self):
        colors = set_plot_style()
        assert isinstance(colors, list)
        assert len(colors) == 5


class TestPlotClassDistribution:
    def test_returns_figure(self):
        df = pd.DataFrame({"level": np.random.randint(0, 5, 50)})
        fig = plot_class_distribution(df, save_folder="test_outputs")
        assert isinstance(fig, plt.Figure)


class TestPlotLearningCurves:
    def test_returns_figure(self):
        history = {
            "train_loss": [0.9, 0.7, 0.5],
            "val_loss": [0.95, 0.75, 0.55],
            "train_acc": [0.5, 0.7, 0.8],
            "val_acc": [0.45, 0.65, 0.75],
        }
        fig = plot_learning_curves(history, save_folder="test_outputs")
        assert isinstance(fig, plt.Figure)


class TestPlotModelDiagnostics:
    def test_returns_figure(self, synthetic_results_df):
        fig = plot_model_diagnostics(synthetic_results_df, save_folder="test_outputs")
        assert isinstance(fig, plt.Figure)


class TestPlotConfidenceAbstention:
    def test_returns_figure(self, synthetic_results_df):
        fig = plot_confidence_abstention_panel(synthetic_results_df, save_folder="test_outputs")
        assert isinstance(fig, plt.Figure)


class TestPlotBinaryClassificationMetrics:
    def test_returns_figure(self, synthetic_results_df):
        fig = plot_binary_classification_metrics(synthetic_results_df, save_folder="test_outputs")
        assert isinstance(fig, plt.Figure)


class TestPlotCalibrationCurves:
    def test_returns_figure(self, synthetic_results_df):
        y_true = synthetic_results_df["true"].values
        prob_matrix = synthetic_results_df[[f"prob_{i}" for i in range(5)]].values
        fig = plot_calibration_curves(y_true, prob_matrix, save_folder="test_outputs")
        assert isinstance(fig, plt.Figure)


class TestPlotErrorDistribution:
    def test_returns_figure_without_confidences(self, synthetic_results_df):
        fig = plot_error_distribution(
            synthetic_results_df["true"].values,
            synthetic_results_df["pred"].values,
            save_folder="test_outputs",
        )
        assert isinstance(fig, plt.Figure)

    def test_returns_figure_with_confidences(self, synthetic_results_df):
        probs = synthetic_results_df[[f"prob_{i}" for i in range(5)]].values
        confidences = np.max(probs, axis=1)
        fig = plot_error_distribution(
            synthetic_results_df["true"].values,
            synthetic_results_df["pred"].values,
            confidences=confidences,
            save_folder="test_outputs",
        )
        assert isinstance(fig, plt.Figure)
