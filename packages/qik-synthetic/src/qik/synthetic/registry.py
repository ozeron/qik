"""Model registry for `SyntheticEngine`."""

from __future__ import annotations

from .errors import SyntheticDependencyError, SyntheticValidationError
from .models import AR1Model, GaussianRandomWalkModel


def _model_from_name(name: str):
    key = name.strip().lower()
    if key == "gaussian_rw":
        return GaussianRandomWalkModel
    if key == "ar1":
        return AR1Model
    if key == "garch":
        try:
            from .models.garch import GARCHModel
            # Validate optional dependencies only when model is selected.
            from .models.garch import _require_arch

            _require_arch()
        except SyntheticDependencyError as exc:
            raise SyntheticDependencyError(
                "Optional dependency missing for garch model. Install qik-synthetic[models]."
            ) from exc
        return GARCHModel
    raise SyntheticValidationError(f"Unknown model family: {name}")


def resolve_model(name: str):
    model_cls = _model_from_name(name)
    return model_cls()


def list_models() -> tuple[str, ...]:
    return ("gaussian_rw", "ar1", "garch")
