"""Load and validate the persisted clause-classifier artifact."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import joblib

from ml.train_classifier import TARGET_CLAUSES

BACKEND_DIR = Path(__file__).resolve().parents[1]
_TRAIN_COMMAND = "python -m ml.train_classifier --cuad-path data/cuad/CUAD_v1.json"


class ModelArtifactError(RuntimeError):
    """Raised when the configured clause-classifier artifact cannot be used."""


def resolve_model_artifact_path(configured_path: str | Path) -> Path:
    """Resolve a configured artifact path, treating relative paths as backend-relative."""
    path = Path(configured_path).expanduser()
    if not path.is_absolute():
        path = BACKEND_DIR / path
    return path.resolve()


def load_classifier_artifact(artifact_path: str | Path) -> Any:
    """Load a trusted joblib pipeline and verify it is compatible with inference."""
    path = Path(artifact_path)
    if not path.is_file():
        raise ModelArtifactError(
            f"Clause classifier artifact not found at {path}. "
            f"Generate it from backend/ with: {_TRAIN_COMMAND}"
        )

    try:
        artifact = joblib.load(path)
    except Exception as exc:  # noqa: BLE001
        raise ModelArtifactError(
            f"Clause classifier artifact at {path} could not be loaded: {exc}. "
            f"Regenerate it from backend/ with: {_TRAIN_COMMAND}"
        ) from exc

    steps = getattr(artifact, "named_steps", None)
    if not isinstance(steps, Mapping) or "tfidf" not in steps or "clf" not in steps:
        raise ModelArtifactError(
            f"Incompatible clause classifier artifact at {path}: expected a fitted "
            "sklearn Pipeline with 'tfidf' and 'clf' steps."
        )
    if not callable(getattr(artifact, "predict", None)) or not callable(
        getattr(artifact, "predict_proba", None)
    ):
        raise ModelArtifactError(
            f"Incompatible clause classifier artifact at {path}: expected both "
            "predict and predict_proba methods."
        )

    classes = getattr(artifact, "classes_", None)
    try:
        class_names = {str(label) for label in classes} if classes is not None else set()
    except TypeError as exc:
        raise ModelArtifactError(
            f"Incompatible clause classifier artifact at {path}: invalid fitted classes."
        ) from exc
    if not class_names:
        raise ModelArtifactError(
            f"Incompatible clause classifier artifact at {path}: no fitted classes found."
        )
    unsupported = sorted(class_names - set(TARGET_CLAUSES))
    if unsupported:
        raise ModelArtifactError(
            f"Incompatible clause classifier artifact at {path}: unsupported classes "
            f"{unsupported}."
        )

    return artifact
