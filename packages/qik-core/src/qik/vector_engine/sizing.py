"""Position sizing: contracts per trade, applied after execution.

With one NETTING position per config, the fills (bars and prices) do not depend on the size, so
sizing is a pass over the finished trade list: each trade's contracts are set from the equity
before it (initial cash + net P&L of the earlier trades of the same config). Two things would make
Nautilus trade a different path, and the engine refuses both instead of guessing:

* a size of 0 contracts (Nautilus would not enter, so every later entry could change), and
* more contracts than the bar's liquidity: on bars Nautilus offers ``max(floor(volume / 4), 1)``
  contracts per open/high/low point and fills the rest one tick worse (docs/vector-engine.md A14).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numba as nb
import numpy as np

from ._checks import is_int
from .account import (
    ERR_INVALID_QTY as ERR_INVALID_QTY,
)
from .account import (
    ERR_NO_STOP as ERR_NO_STOP,
)
from .account import (
    ERR_NONE,
    MODE_CONTRACTS,
    MODE_NOTIONAL,
    MODE_QTY_FEATURE,
    MODE_RISK_FRACTION,
    MODE_RISK_USD,
)
from .account import (
    ERR_QTY_RANGE as ERR_QTY_RANGE,
)
from .account import (
    ERR_UNAFFORDABLE as ERR_UNAFFORDABLE,
)
from .account import (
    ERR_ZERO as ERR_ZERO,
)
from .account import (
    quantity as _contracts_for,
)

__all__ = ["Sizing"]


@dataclass(frozen=True)
class Sizing[Quantity: int | float = int]:
    """Contracts per entry; set exactly one of ``contracts``, ``risk_usd``, ``risk_fraction`` or
    ``notional_usd`` or ``qty_feature``.

    * ``contracts``: the same number for every trade.
    * ``risk_usd``: ``floor(risk_usd / (|entry - stop| * point_value))``; needs a stop on every entry.
    * ``risk_fraction``: as ``risk_usd`` with ``risk_usd = risk_fraction * equity`` before the trade
      (compounding); needs ``initial_cash``.
    * ``notional_usd``: ``floor(notional_usd / (|entry fill| * point_value))``, no stop. The size
      comes from the actual accepted fill price and is the same for every trade (a loss does not
      change the next entry). ``affordability`` is mandatory with it: ``"refuse"`` (needs
      ``initial_cash``) refuses an entry whose notional plus the entry fill's costs exceeds the
      balance before the trade, so fees are never borrowed; ``"unchecked"`` assumes no account
      (independent fixed-notional books). ``min_qty_policy="clamp"`` is refused: it would exceed
      the fixed notional.

    * ``qty_feature``: wanted quantity from the named strategy feature at the entry's
      decision bar, retained through next-open, resting and chase fills. Floors to the lot
      step, caps at ``max_contracts`` and refuses invalid/below-minimum quantities; no clamp.

    The stop distance is in points from the entry fill; costs are not part of the risk.
    ``max_contracts`` caps the computed modes. Computed quantities floor to ``lot_step`` multiples
    and refuse below ``min_qty`` by default. Opt-in ``min_qty_policy="clamp"``
    raises risk quantities to the grid-aligned minimum, potentially exceeding the risk budget.
    Fixed quantities must already be on that grid.
    Defaults retain integer-only construction; explicit fractional steps admit float quantities.
    """

    contracts: Quantity | None = None
    risk_usd: float | None = None
    risk_fraction: float | None = None
    max_contracts: Quantity | None = None

    lot_step: float = 1.0
    min_qty: float = 1.0
    min_qty_policy: Literal["refuse", "clamp"] = "refuse"

    notional_usd: float | None = None
    affordability: Literal["refuse", "unchecked"] | None = None
    qty_feature: str | None = None

    def __post_init__(self) -> None:
        if self.min_qty_policy not in ("refuse", "clamp"):
            raise ValueError("min_qty_policy must be refuse or clamp")
        set_ = [
            x is not None
            for x in (
                self.contracts,
                self.risk_usd,
                self.risk_fraction,
                self.notional_usd,
                self.qty_feature,
            )
        ]
        if sum(set_) != 1:
            raise ValueError(
                "Sizing needs exactly one of contracts, risk_usd, risk_fraction, notional_usd, qty_feature"
            )
        if self.qty_feature is not None:
            if not isinstance(self.qty_feature, str) or not self.qty_feature.strip():
                raise ValueError("qty_feature must name a strategy feature lane")
            if self.min_qty_policy == "clamp":
                raise ValueError("min_qty_policy='clamp' is not supported with qty_feature")
        if self.notional_usd is None:
            if self.affordability is not None:
                raise ValueError("affordability applies to notional_usd only")
        else:
            if self.affordability not in ("refuse", "unchecked"):
                raise ValueError(
                    "Sizing(notional_usd=...) needs affordability='refuse' or 'unchecked'"
                )
            if isinstance(self.notional_usd, bool) or not (
                np.isfinite(self.notional_usd) and self.notional_usd > 0
            ):
                raise ValueError(
                    f"notional_usd must be finite and positive, got {self.notional_usd!r}"
                )
            if self.min_qty_policy == "clamp":
                raise ValueError("min_qty_policy='clamp' would exceed a fixed notional")
        for name in ("lot_step", "min_qty"):
            v = getattr(self, name)
            if isinstance(v, bool) or not (np.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be finite and positive")
        if self.min_qty_policy == "clamp" and (
            abs(self.min_qty / self.lot_step - round(self.min_qty / self.lot_step)) > 1e-7
        ):
            raise ValueError("clamped min_qty must be a multiple of lot_step")
        for name in ("contracts", "max_contracts"):
            v = getattr(self, name)
            if v is not None and (
                isinstance(v, bool)
                or (self.lot_step == 1 and self.min_qty == 1 and not is_int(v))
                or not np.isfinite(v)
                or v < self.min_qty
                or abs(v / self.lot_step - round(v / self.lot_step)) > 1e-7
            ):
                raise ValueError(
                    f"{name} must be a positive integer multiple of lot_step and >= min_qty"
                )
        if self.risk_usd is not None and not (np.isfinite(self.risk_usd) and self.risk_usd > 0):
            raise ValueError(f"risk_usd must be positive, got {self.risk_usd!r}")
        if self.risk_fraction is not None and not 0 < self.risk_fraction <= 1:
            raise ValueError(f"risk_fraction must be in (0, 1], got {self.risk_fraction!r}")

    @property
    def mode(self) -> int:
        if self.qty_feature is not None:
            return MODE_QTY_FEATURE
        if self.contracts is not None:
            return MODE_CONTRACTS
        if self.notional_usd is not None:
            return MODE_NOTIONAL
        return MODE_RISK_USD if self.risk_usd is not None else MODE_RISK_FRACTION

    @property
    def amount(self) -> float:
        if self.qty_feature is not None:
            return 0.0  # the amount is supplied by the decision-bar feature lane
        for x in (self.contracts, self.risk_usd, self.risk_fraction, self.notional_usd):
            if x is not None:
                return float(x)
        raise AssertionError("unreachable: validated in __post_init__")

    @property
    def check_cash(self) -> bool:
        return self.affordability == "refuse"

    @property
    def cap(self) -> float:
        return float(self.max_contracts) if self.max_contracts is not None else 0


@nb.njit(cache=True)
def size_trades(
    cfg: np.ndarray,
    net_per_contract: np.ndarray,
    risk_per_contract: np.ndarray,
    notional_per_unit: np.ndarray,
    entry_cost_per_unit: np.ndarray,
    mode: int,
    amount: float,
    cap: float,
    initial_cash: float,
    k: int,
    lot_step: float = 1.0,
    min_qty: float = 1.0,
    clamp_min_qty: bool = False,
    check_cash: bool = False,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Sequential sizing of closed trades grouped by config, in time order within each config.

    ``notional_per_unit`` and ``entry_cost_per_unit`` (per trade, from the entry fill) matter to
    ``MODE_NOTIONAL`` only. Returns ``(contracts, final_equity[k], err, err_row)``.
    """

    m = cfg.shape[0]
    contracts = np.zeros(m, np.float64)
    final_equity = np.full(k, initial_cash)
    for r in range(m):
        j = cfg[r]
        n = _contracts_for(
            mode,
            amount,
            cap,
            final_equity[j],
            risk_per_contract[r],
            lot_step,
            min_qty,
            clamp_min_qty,
            notional_per_unit[r],
            entry_cost_per_unit[r],
            check_cash,
        )
        if n < 0:
            return contracts, final_equity, int(-n), r
        contracts[r] = n
        final_equity[j] += n * net_per_contract[r]
    return contracts, final_equity, ERR_NONE, -1


@nb.njit(cache=True)
def size_open(
    cfg: np.ndarray,
    equity: np.ndarray,
    risk_per_contract: np.ndarray,
    notional_per_unit: np.ndarray,
    entry_cost_per_unit: np.ndarray,
    mode: int,
    amount: float,
    cap: float,
    lot_step: float = 1.0,
    min_qty: float = 1.0,
    clamp_min_qty: bool = False,
    check_cash: bool = False,
) -> np.ndarray:
    """Contracts of the positions still open after the last bar (equity = after the last close)."""

    m = cfg.shape[0]
    out = np.zeros(m, np.float64)
    for r in range(m):
        out[r] = _contracts_for(
            mode,
            amount,
            cap,
            equity[cfg[r]],
            risk_per_contract[r],
            lot_step,
            min_qty,
            clamp_min_qty,
            notional_per_unit[r],
            entry_cost_per_unit[r],
            check_cash,
        )  # < 0: error code
    return out


def point_liquidity(volume: np.ndarray, lot_step: float = 1.0, min_qty: float = 1.0) -> np.ndarray:
    """Contracts Nautilus fills at one price at each open/high/low point of a bar (A14)."""

    return np.maximum(np.floor(volume / 4.0 / lot_step + 1e-10) * lot_step, max(1.0, min_qty))
