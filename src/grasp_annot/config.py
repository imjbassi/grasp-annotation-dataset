"""Centralised configuration.

All runtime configuration is sourced from environment variables (optionally
loaded from a local ``.env`` file) so that nothing secret is ever committed and
the same code runs unchanged in CI, locally, or in a container.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

try:
    # Optional: load a local .env if python-dotenv is installed.
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover - dotenv is a convenience, not required.
    pass


# Repository layout ----------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
FRAMES_DIR = DATA_DIR / "frames"
EXPORTS_DIR = DATA_DIR / "exports"


@dataclass(frozen=True)
class Settings:
    """Resolved settings for a run."""

    api_key: str
    project_name: str
    dataset_name: str
    ontology_name: str
    batch_name: str

    @staticmethod
    def from_env() -> "Settings":
        api_key = os.getenv("LABELBOX_API_KEY", "").strip()
        if not api_key:
            raise EnvironmentError(
                "LABELBOX_API_KEY is not set. Copy .env.example to .env and add "
                "your key, or `export LABELBOX_API_KEY=...`. Get a free key at "
                "https://app.labelbox.com/ under Account → API keys."
            )
        return Settings(
            api_key=api_key,
            project_name=os.getenv("LB_PROJECT_NAME", "robot-grasp-annotation"),
            dataset_name=os.getenv("LB_DATASET_NAME", "grasp-frames"),
            ontology_name=os.getenv("LB_ONTOLOGY_NAME", "grasp-ontology-v1"),
            batch_name=os.getenv("LB_BATCH_NAME", "grasp-batch-001"),
        )


def ensure_dirs() -> None:
    """Create the data directories if they do not already exist."""
    for d in (RAW_DIR, FRAMES_DIR, EXPORTS_DIR):
        d.mkdir(parents=True, exist_ok=True)
