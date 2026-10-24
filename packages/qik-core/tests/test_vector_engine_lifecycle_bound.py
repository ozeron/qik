"""Exercise expiry with callbacks bound through all three scheduler paths."""

from __future__ import annotations

import numba as nb
import pandas as pd
import pytest
from test_vector_engine_lifecycle import tape
from vector_engine.lifecycle_limit_fixture import F, limit_expiry_step

from qik.vector_engine import Rules, run
from qik.vector_engine.testing import assert_python_jit_equal


@pytest.mark.parametrize("latency", [0, 1])
def test_bound_limit_expiry_before_callback(latency: int) -> None:
    bars = tape(pd.date_range("2023-01-03 18:00", periods=4, freq="D", tz="America/New_York"))
    kw = dict(
        features=F(),
        rules=Rules(day_order_expiry="cancel", latency_bars=latency),
        signal_fields=("entry_side", "entry_limit", "stop"),
    )
    result = run(limit_expiry_step, bars, **kw)
    assert result.trades.empty and result.open_positions.empty
    assert result.state().pending_seen == 0
    if not nb.config.DISABLE_JIT:
        assert_python_jit_equal(limit_expiry_step, bars, **kw)
