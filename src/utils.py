from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import tensorflow as tf


@dataclass(frozen=True)
class Paths:
    project_root: Path
    outputs_dir: Path
    models_dir: Path
    figures_dir: Path

    @staticmethod
    def from_root(project_root: Path) -> "Paths":
        outputs_dir = project_root / "outputs"
        models_dir = outputs_dir / "models"
        figures_dir = outputs_dir / "figures"
        models_dir.mkdir(parents=True, exist_ok=True)
        figures_dir.mkdir(parents=True, exist_ok=True)
        return Paths(
            project_root=project_root,
            outputs_dir=outputs_dir,
            models_dir=models_dir,
            figures_dir=figures_dir,
        )


def set_reproducibility(seed: int = 42) -> None:
    """Best-effort reproducibility."""
    tf.keras.utils.set_random_seed(seed)
    np.random.seed(seed)


def gpu_available() -> bool:
    return len(tf.config.list_physical_devices("GPU")) > 0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
