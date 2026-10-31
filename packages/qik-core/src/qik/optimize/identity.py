"""Bind numerical evidence to its pair, full family, endpoint and dated calendar.

The numerical kernels remain in inference/unit. Bare arrays are useful for
simulation; campaign decisions require the dated panel entry points below.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from typing import Literal, TypedDict, Unpack

import numpy as np

from qik.optimize import inference, unit
from qik.optimize.artifacts import SessionPanel, validate_dates
from qik.optimize.inference import MaxStatistic
from qik.optimize.unit import UnitScore


class BootstrapOptions(TypedDict, total=False):
    block: int
    draws: int
    seed: int


class UnitOptions(BootstrapOptions, total=False):
    parking: tuple[np.ndarray, np.ndarray] | None


def array_digest(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for values in arrays:
        canonical = np.ascontiguousarray(values, dtype="<f8")
        digest.update(str(canonical.shape).encode())
        digest.update(canonical.tobytes())
    return digest.hexdigest()


def bind_test(
    result: MaxStatistic,
    pair_ids: tuple[str, ...],
    endpoint: str,
) -> MaxStatistic:
    if len(pair_ids) != len(result.effects) or not pair_ids:
        raise ValueError("full-family pair identities required")
    family_id = hashlib.sha256((endpoint + "|" + "|".join(pair_ids)).encode()).hexdigest()
    return replace(result, family_id=family_id, pair_ids=pair_ids, endpoint=endpoint)


def score_unit(
    baseline: np.ndarray, candidate: np.ndarray, **kwargs: Unpack[UnitOptions]
) -> UnitScore:
    parking = kwargs.get("parking")
    if parking is not None:
        kwargs["parking"] = (np.asarray(parking[0]), np.asarray(parking[1]))
    inputs = (
        (baseline, candidate)
        if parking is None
        else (baseline, candidate, np.asarray(parking[0]), np.asarray(parking[1]))
    )
    return replace(unit.score_unit(baseline, candidate, **kwargs), pair_id=array_digest(*inputs))


def incremental_max_test(
    baseline: np.ndarray, candidates: np.ndarray, **kwargs: Unpack[BootstrapOptions]
) -> MaxStatistic:
    result = inference.incremental_max_test(baseline, candidates, **kwargs)
    bases = (
        np.repeat(np.asarray(baseline)[:, None], candidates.shape[1], axis=1)
        if np.ndim(baseline) == 1
        else baseline
    )
    pairs = tuple(array_digest(b, c) for b, c in zip(bases.T, candidates.T, strict=True))
    return bind_test(result, pairs, "sharpe")


def endpoint_max_test(
    baseline: np.ndarray,
    candidates: np.ndarray,
    *,
    endpoint: unit.RiskEndpoint,
    **kwargs: Unpack[BootstrapOptions],
) -> MaxStatistic:
    result = unit.endpoint_max_test(baseline, candidates, endpoint=endpoint, **kwargs)
    pairs = tuple(array_digest(baseline, c) for c in candidates.T)
    return bind_test(result, pairs, endpoint)


def _calendar(panel: SessionPanel, *, unsealed: bool) -> str:
    dates = validate_dates([d.isoformat() for d in panel.dates], unsealed=unsealed)
    if len(dates) != len(panel.values) or tuple(sorted(set(dates))) != dates:
        raise ValueError("invalid dated panel")
    return hashlib.sha256("|".join(d.isoformat() for d in dates).encode()).hexdigest()


def score_panel(
    panel: SessionPanel,
    baseline: str,
    candidate: str,
    *,
    unsealed: bool = False,
    parking_columns: tuple[str, str] | None = None,
    **kwargs: Unpack[UnitOptions],
) -> UnitScore:
    calendar_id = _calendar(panel, unsealed=unsealed)
    if parking_columns is not None:
        if kwargs.get("parking") is not None:
            raise ValueError("declare parking columns or arrays, not both")
        kwargs["parking"] = (panel.column(parking_columns[0]), panel.column(parking_columns[1]))
    result = score_unit(panel.column(baseline), panel.column(candidate), **kwargs)
    return replace(
        result,
        pair_id=hashlib.sha256((calendar_id + result.pair_id).encode()).hexdigest(),
        calendar_id=calendar_id,
    )


def test_panel(
    panel: SessionPanel,
    baseline: str,
    candidates: tuple[str, ...],
    *,
    endpoint: Literal["sharpe", "es5", "drawdown", "parking"] = "sharpe",
    unsealed: bool = False,
    parking_columns: tuple[str, ...] | None = None,
    **kwargs: Unpack[BootstrapOptions],
) -> MaxStatistic:
    calendar_id = _calendar(panel, unsealed=unsealed)
    if not candidates or len(set(candidates)) != len(candidates):
        raise ValueError("unique full-family candidate columns required")
    base = panel.column(baseline)
    matrix = np.column_stack([panel.column(c) for c in candidates])
    if parking_columns is not None and len(parking_columns) != len(candidates) + 1:
        raise ValueError("parking family must include baseline plus every candidate")
    if endpoint == "parking":
        if parking_columns is None:
            raise ValueError("parking endpoint needs its full aligned indicator family")
        result = endpoint_max_test(
            panel.column(parking_columns[0]),
            np.column_stack([panel.column(c) for c in parking_columns[1:]]),
            endpoint="parking",
            **kwargs,
        )
    else:
        result = (
            incremental_max_test(base, matrix, **kwargs)
            if endpoint == "sharpe"
            else endpoint_max_test(base, matrix, endpoint=endpoint, **kwargs)
        )
    if parking_columns is not None:
        raw_pairs = tuple(
            array_digest(base, panel.column(c), panel.column(parking_columns[0]), panel.column(p))
            for c, p in zip(candidates, parking_columns[1:], strict=True)
        )
    else:
        raw_pairs = tuple(array_digest(base, panel.column(c)) for c in candidates)
    pairs = tuple(hashlib.sha256((calendar_id + p).encode()).hexdigest() for p in raw_pairs)
    return bind_test(result, pairs, endpoint)


def score_stress_panel(
    reference: UnitScore,
    panel: SessionPanel,
    baseline: str,
    candidate: str,
    *,
    unsealed: bool = False,
    **kwargs: Unpack[BootstrapOptions],
) -> UnitScore:
    """G5 paired mean NI at the ORIGINAL G1–G3 dollar margin, not a new margin.

    The stress tape must come from the actual shared execution replay. This
    statistic is G5 evidence and cannot substitute for primary G3 evidence.
    """
    if reference.ni_margin <= 0 or not reference.pair_id:
        raise ValueError("positive original NI margin and bound reference required")
    score = score_panel(panel, baseline, candidate, unsealed=unsealed, **kwargs)
    indices = inference.circular_indices(len(panel.dates), score.block, score.draws, score.seed)
    means = (panel.column(candidate) - panel.column(baseline))[indices].mean(axis=1)
    return replace(
        score,
        ni_margin=reference.ni_margin,
        p_mean_ni=float((means > -reference.ni_margin).mean()),
        stress_reference_id=reference.pair_id,
        pair_id=hashlib.sha256(
            (reference.pair_id + score.pair_id + repr(reference.ni_margin)).encode()
        ).hexdigest(),
    )
