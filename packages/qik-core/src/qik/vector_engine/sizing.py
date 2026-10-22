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

__all__ = ["Sizing"]

MODE_CONTRACTS = 0
MODE_RISK_USD = 1
MODE_RISK_FRACTION = 2

ERR_NONE = 0
ERR_ZERO = 1
ERR_NO_STOP = 2


@dataclass(frozen=True)
class Sizing:
    """Contracts per entry; set exactly one of ``contracts``, ``risk_usd`` or ``risk_fraction``.

    * ``contracts``: the same number for every trade.
    * ``risk_usd``: ``floor(risk_usd / (|entry - stop| * point_value))``; needs a stop on every entry.
    * ``risk_fraction``: as ``risk_usd`` with ``risk_usd = risk_fraction * equity`` before the trade
      (compounding); needs ``initial_cash``.

    The stop distance is in points from the entry fill; costs are not part of the risk.
    ``max_contracts`` caps the risk modes.
    """

    contracts: int | None = None
    risk_usd: float | None = None
    risk_fraction: float | None = None
    max_contracts: int | None = None

    def __post_init__(self) -> None:
        set_ = [x is not None for x in (self.contracts, self.risk_usd, self.risk_fraction)]
        if sum(set_) != 1:
            raise ValueError("Sizing needs exactly one of contracts, risk_usd, risk_fraction")
        if self.contracts is not None and (
            int(self.contracts) != self.contracts or self.contracts < 1
        ):
            raise ValueError(f"contracts must be a positive integer, got {self.contracts!r}")
        if self.risk_usd is not None and not self.risk_usd > 0:
            raise ValueError(f"risk_usd must be positive, got {self.risk_usd!r}")
        if self.risk_fraction is not None and not 0 < self.risk_fraction <= 1:
            raise ValueError(f"risk_fraction must be in (0, 1], got {self.risk_fraction!r}")
        if self.max_contracts is not None and (
            int(self.max_contracts) != self.max_contracts or self.max_contracts < 1
        ):
            raise ValueError(
                f"max_contracts must be a positive integer, got {self.max_contracts!r}"
            )

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
    def cap(self) -> int:
        return int(self.max_contracts) if self.max_contracts is not None else 0


@nb.njit(cache=True)
def _one(mode, amount, cap, equity, risk_per_contract):
    """Contracts for one entry, or a negative error code."""

    if mode == MODE_CONTRACTS:
        return np.int64(amount)
    if not (risk_per_contract > 0.0):  # NaN stop
        return -ERR_NO_STOP
    budget = amount if mode == MODE_RISK_USD else amount * equity
    if not (budget > 0.0):
        return -ERR_ZERO
    n = np.int64(np.floor(budget / risk_per_contract + 1e-9))
    if cap > 0 and n > cap:
        n = np.int64(cap)
    if n < 1:
        return -ERR_ZERO
    return n


@nb.njit(cache=True)
def size_trades(cfg, net_per_contract, risk_per_contract, mode, amount, cap, initial_cash, k):
    """Sequential sizing of closed trades grouped by config, in time order within each config.

    Returns ``(contracts, final_equity[k], err, err_row)``.
    """

    m = cfg.shape[0]
    contracts = np.zeros(m, np.int64)
    final_equity = np.full(k, initial_cash)
    for r in range(m):
        j = cfg[r]
        n = _one(mode, amount, cap, final_equity[j], risk_per_contract[r])
        if n < 0:
            return contracts, final_equity, -n, r
        contracts[r] = n
        final_equity[j] += n * net_per_contract[r]
    return contracts, final_equity, ERR_NONE, -1


@nb.njit(cache=True)
def size_open(cfg, equity, risk_per_contract, mode, amount, cap):
    """Contracts of the positions still open after the last bar (equity = after the last close)."""

    m = cfg.shape[0]
    out = np.zeros(m, np.int64)
    for r in range(m):
        out[r] = _one(mode, amount, cap, equity[cfg[r]], risk_per_contract[r])  # < 0: error code
    return out


def point_liquidity(volume: np.ndarray) -> np.ndarray:
    """Contracts Nautilus fills at one price at each open/high/low point of a bar (A14)."""

    return np.maximum(np.floor(volume / 4.0), 1.0)
