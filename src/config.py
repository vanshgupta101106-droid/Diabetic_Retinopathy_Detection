"""
Project Configuration
=====================

Central place for path constants and YAML config loading.
All values are side-effect-free — no directories are created at import time.
"""

from pathlib import Path

import yaml

# Resolve root relative to this file (src/config.py -> project_root)
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# --- Path constants (no side effects) ---
DATA_DIR = PROJECT_ROOT / "data"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_DIR = PROJECT_ROOT / "models"


def load_config() -> dict:
    """Load and return the default YAML configuration.

    Returns:
        Parsed config dictionary.
    """
    config_path = PROJECT_ROOT / "configs" / "default.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def _ensure_dirs() -> None:
    """Create output directories if they don't exist.

    Called by entry-points (training, scripts) rather than at import time.
    """
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Quality threshold loaded from YAML (kept as module-level for backward compat)
QUALITY_THRESHOLD: float = load_config()["quality"]["threshold"]
