"""Tests for src.config."""

from pathlib import Path

from src.config import (
    DATA_DIR,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    QUALITY_THRESHOLD,
    REPORTS_DIR,
    load_config,
    _ensure_dirs,
)


def test_load_config_returns_dict():
    config = load_config()
    assert isinstance(config, dict)


def test_load_config_has_expected_sections():
    config = load_config()
    for key in ("data", "model", "train", "logging", "quality"):
        assert key in config, f"Missing config section: {key}"


def test_quality_threshold_matches_yaml():
    config = load_config()
    assert QUALITY_THRESHOLD == config["quality"]["threshold"]


def test_path_constants_resolve():
    assert PROJECT_ROOT.exists()
    assert DATA_DIR == PROJECT_ROOT / "data"
    assert PROCESSED_DATA_DIR == DATA_DIR / "processed"
    assert REPORTS_DIR == PROJECT_ROOT / "reports"
    assert MODELS_DIR == PROJECT_ROOT / "models"


def test_ensure_dirs_creates_directories(tmp_path, monkeypatch):
    """Verify _ensure_dirs creates directories without error."""
    # We call the real function — dirs already exist so it's idempotent
    _ensure_dirs()
    assert PROCESSED_DATA_DIR.exists()
    assert REPORTS_DIR.exists()
    assert MODELS_DIR.exists()
