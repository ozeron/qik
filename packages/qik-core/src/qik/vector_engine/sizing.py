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

import numba as nb
import numpy as np

from ._checks import is_int
from .account import (
    ERR_NO_STOP as ERR_NO_STOP,
)
from .account import (
    ERR_NONE,
    MODE_CONTRACTS,
    MODE_RISK_FRACTION,
    MODE_RISK_USD,
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
    """Contracts per entry; set exactly one of ``contracts``, ``risk_usd`` or ``risk_fraction``.

    * ``contracts``: the same number for every trade.
    * ``risk_usd``: ``floor(risk_usd / (|entry - stop| * point_value))``; needs a stop on every entry.
    * ``risk_fraction``: as ``risk_usd`` with ``risk_usd = risk_fraction * equity`` before the trade
      (compounding); needs ``initial_cash``.

    The stop distance is in points from the entry fill; costs are not part of the risk.
    ``max_contracts`` caps the risk modes. Risk quantities floor to ``lot_step`` multiples
    and refuse below ``min_qty``; fixed quantities must already be on that grid.
    Defaults retain integer-only construction; explicit fractional steps admit float quantities.
    """

    contracts: Quantity | None = None
    risk_usd: float | None = None
    risk_fraction: float | None = None
    max_contracts: Quantity | None = None

    lot_step: float = 1.0
    min_qty: float = 1.0

    def __post_init__(self) -> None:
        set_ = [x is not None for x in (self.contracts, self.risk_usd, self.risk_fraction)]
        if sum(set_) != 1:
            raise ValueError("Sizing needs exactly one of contracts, risk_usd, risk_fraction")
        for name in ("lot_step", "min_qty"):
            v = getattr(self, name)
            if isinstance(v, bool) or not (np.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be finite and positive")
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
        if self.contracts is not None:
            return MODE_CONTRACTS
        return MODE_RISK_USD if self.risk_usd is not None else MODE_RISK_FRACTION

    @property
    def amount(self) -> float:
        for x in (self.contracts, self.risk_usd, self.risk_fraction):
            if x is not None:
                return float(x)
        raise AssertionError("unreachable: validated in __post_init__")

    @property
    def cap(self) -> float:
        return float(self.max_contracts) if self.max_contracts is not None else 0


@nb.njit(cache=True)
def size_trades(
    cfg: np.ndarray,
    net_per_contract: np.ndarray,
    risk_per_contract: np.ndarray,
    mode: int,
    amount: float,
    cap: float,
    initial_cash: float,
    k: int,
    lot_step: float = 1.0,
    min_qty: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Sequential sizing of closed trades grouped by config, in time order within each config.

    Returns ``(contracts, final_equity[k], err, err_row)``.
    """

    m = cfg.shape[0]
    contracts = np.zeros(m, np.float64)
    final_equity = np.full(k, initial_cash)
    for r in range(m):
        j = cfg[r]
        n = _contracts_for(
            mode, amount, cap, final_equity[j], risk_per_contract[r], lot_step, min_qty
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
    mode: int,
    amount: float,
    cap: float,
    lot_step: float = 1.0,
    min_qty: float = 1.0,
) -> np.ndarray:
    """Contracts of the positions still open after the last bar (equity = after the last close)."""

    m = cfg.shape[0]
    out = np.zeros(m, np.float64)
    for r in range(m):
        out[r] = _contracts_for(
            mode, amount, cap, equity[cfg[r]], risk_per_contract[r], lot_step, min_qty
        )  # < 0: error code
    return out


def point_liquidity(volume: np.ndarray, lot_step: float = 1.0, min_qty: float = 1.0) -> np.ndarray:
    """Contracts Nautilus fills at one price at each open/high/low point of a bar (A14)."""

    return np.maximum(np.floor(volume / 4.0 / lot_step + 1e-10) * lot_step, max(1.0, min_qty))
