"""Model registry for `SyntheticEngine`."""

from __future__ import annotations

from .errors import SyntheticDependencyError, SyntheticValidationError
from .models import (
    AR1Model,
    EmpiricalSVModel,
    GaussianRandomWalkModel,
    RegimeConditionedBootstrapModel,
    StationaryBootstrapModel,
    StudentTRandomWalkModel,
)


def _model_from_name(name: str):
    key = name.strip().lower()
    if key == "gaussian_rw":
        return GaussianRandomWalkModel
    if key == "ar1":
        return AR1Model
    if key in {"stationary_bootstrap", "empirical_bootstrap"}:
        return StationaryBootstrapModel
    if key in {"empirical_sv", "regime_bootstrap", "seasonal_bootstrap"}:
        return EmpiricalSVModel
    if key == "regime_conditioned_bootstrap":
        return RegimeConditionedBootstrapModel
    if key in {"student_t_rw", "student_t", "realistic_btc_1h", "recommended_realism"}:
        return StudentTRandomWalkModel
    if key == "garch":
        try:
            from .models.garch import GARCHModel
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
    return (
        "gaussian_rw",
        "ar1",
        "garch",
        "stationary_bootstrap",
        "student_t_rw",
        "empirical_sv",
        "regime_conditioned_bootstrap",
        "realistic_btc_1h",
    )
