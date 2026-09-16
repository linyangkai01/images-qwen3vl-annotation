"""Thin helpers for loading hazard-annotation project configs.

The authoritative configuration object is :class:`traffic_cv_closed_loop.config.ProjectConfig`.
This module only adds the two-tier (quick/full) profile overlay and run-mode
resolution used by the construction-hazard entry point.
"""

from __future__ import annotations

from pathlib import Path

from traffic_cv_closed_loop.config import ProjectConfig

VALID_RUN_MODES = ("quick", "full")
DEFAULT_RUN_MODE = "quick"


def normalize_run_mode(mode: str | None) -> str:
    """Return a valid run mode, falling back to the default when unknown."""
    value = str(mode or DEFAULT_RUN_MODE).strip().lower()
    if value not in VALID_RUN_MODES:
        return DEFAULT_RUN_MODE
    return value


def load_hazard_config(config_path: str | Path, mode: str | None = None) -> ProjectConfig:
    """Load a project config and overlay the requested (or configured) profile.

    ``mode`` (from ``--mode``) wins over the ``run_mode`` field stored in the
    config file, which in turn falls back to ``quick``.
    """
    config = ProjectConfig.load(config_path)
    selected = normalize_run_mode(mode or config.run_mode)
    config.apply_profile(selected)
    return config
