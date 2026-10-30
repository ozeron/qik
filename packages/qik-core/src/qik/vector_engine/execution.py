"""Frozen account policy records for the single vector engine.

Only the implemented default accounting policy is accepted. Declarations are
validated before planning, including runs with no bars; no policy selects codegen.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AccountSpec:
    model: str = "pnl_balance"
    currency: str = "USD"
    risk_equity: str = "settled"
    target_equity: str = "marked"


@dataclass(frozen=True)
class AdmissionSpec:
    size_at: str = "fill"


@dataclass(frozen=True)
class ExecutionSpec:
    account: AccountSpec | None = None
    admission: AdmissionSpec | None = None


def normalize_execution(execution: ExecutionSpec | None) -> None:
    """Refuse declarations the engine or its V2 host cannot honor."""
    if execution is None:
        return
    if not isinstance(execution, ExecutionSpec):
        raise TypeError("execution must be ExecutionSpec or None")
    if execution.account is not None and (
        not isinstance(execution.account, AccountSpec) or execution.account != AccountSpec()
    ):
        raise ValueError(f"unsupported account policy: {execution.account!r}")
    if execution.admission is not None and (
        not isinstance(execution.admission, AdmissionSpec) or execution.admission != AdmissionSpec()
    ):
        raise ValueError(f"unsupported account policy: {execution.admission!r}")
