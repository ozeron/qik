"""Synthetic domain errors."""


class SyntheticError(RuntimeError):
    """Base exception for all qik-synthetic failures."""


class SyntheticValidationError(SyntheticError, ValueError):
    """Raised for invalid requests, context, or generated payloads."""


class SyntheticContextError(SyntheticError, ValueError):
    """Raised when generation context conflicts with model assumptions."""


class SyntheticConversionError(SyntheticValidationError):
    """Raised when backtesting handoff data cannot be converted safely."""


class SyntheticDependencyError(SyntheticError, RuntimeError):
    """Raised when optional runtime dependencies are required but unavailable."""
