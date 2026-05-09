"""Built-in synthetic models."""

from .gaussian import GaussianRandomWalkModel
from .ar1 import AR1Model
from .bootstrap import StationaryBootstrapModel
from .student_t import StudentTRandomWalkModel

__all__ = [
    "GaussianRandomWalkModel",
    "AR1Model",
    "StationaryBootstrapModel",
    "StudentTRandomWalkModel",
]
