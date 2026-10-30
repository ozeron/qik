"""Internal decision columns shared by typed capture and validation.

This transport carries declaration masks via absent optional arrays; it is not an
execution or strategy authoring API. Matching stays in the bound kernel runner.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .api import Bars, Costs, Instrument, Result, Rules
    from .sizing import Sizing


@dataclass(frozen=True)
class DecisionColumns:
    entry_side: np.ndarray
    exit: np.ndarray
    stop: np.ndarray | None = None
    target: np.ndarray | None = None
    entry_limit: np.ndarray | None = None
    stop_points: np.ndarray | None = None
    target_points: np.ndarray | None = None
    entry_stop: np.ndarray | None = None
    stop_update: np.ndarray | None = None
    target_update: np.ndarray | None = None
    breakeven_after_points: np.ndarray | None = None
    breakeven_offset_points: np.ndarray | None = None
    flatten: np.ndarray | None = None
    exit_long: np.ndarray | None = None
    exit_short: np.ndarray | None = None
    entry_limit_ioc: np.ndarray | None = None
    entry_stop_long: np.ndarray | None = None
    entry_limit_ioc_long: np.ndarray | None = None
    stop_long: np.ndarray | None = None
    target_long: np.ndarray | None = None
    stop_points_long: np.ndarray | None = None
    target_points_long: np.ndarray | None = None
    entry_stop_short: np.ndarray | None = None
    entry_limit_ioc_short: np.ndarray | None = None
    stop_short: np.ndarray | None = None
    target_short: np.ndarray | None = None
    stop_points_short: np.ndarray | None = None
    target_points_short: np.ndarray | None = None
    entry_from_open: np.ndarray | None = None
    stop_fraction: np.ndarray | None = None
    target_fraction: np.ndarray | None = None

    def evaluate(
        self,
        bars: Bars,
        *,
        rules: Rules,
        instrument: Instrument | None,
        costs: Costs | None,
        sizing: Sizing[int] | Sizing[float] | None,
        initial_cash: float | None,
        _run_columns: Callable[..., tuple],
        _entry_quantities: np.ndarray | None = None,
        _loop_quantities: np.ndarray | None = None,
        _record_rejections: bool = False,
    ) -> Result:
        """Apply shared checks/accounting to a validated or already-matched column.

        A callback is mandatory: this carrier never selects an array executor.
        Replay supplies its validation sentinel; typed runs supply matched buffers.
        """
        from .api import _simulate

        return _simulate(
            bars,
            self,
            rules=rules,
            instrument=instrument,
            costs=costs,
            sizing=sizing,
            initial_cash=initial_cash,
            _run_columns=_run_columns,
            _entry_quantities=_entry_quantities,
            _loop_quantities=_loop_quantities,
            _record_rejections=_record_rejections,
        )
