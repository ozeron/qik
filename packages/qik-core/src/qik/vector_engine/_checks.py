"""Small input checks shared by the vector engine's modules."""

from __future__ import annotations

import numpy as np


def is_int(x: object) -> bool:
    """A real integer (``int`` or numpy integer), not ``bool`` and not ``2.0``."""

    return isinstance(x, (int, np.integer)) and not isinstance(x, (bool, np.bool_))
