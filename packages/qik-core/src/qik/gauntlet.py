"""Fixed robustness gauntlet: one fast, fail-fast pass over a strategy's robustness checks.

A strategy enters as a ``Subject``: its config (a frozen dataclass), a ``run(cfg, data) -> trades``
function, its raw OHLCV tape and its eligible sessions. Every check is scored on the same statistic:
net P&L and the annualized Sharpe of eligible-session P&L (no-trade sessions count as 0).

Checks, cheapest first (``run_gauntlet`` stops at the first FAIL unless ``fail_fast=False``):

==============  ====================================================  ==========================
check           question                                              cost (strategy runs)
==============  ====================================================  ==========================
base            is there anything to validate (net > 0, Sharpe > 0)    1
stress          does it survive costs x1.5 / +1 tick / one bar late    1 per variant
trade_mc        block-bootstrap of session P&L: P(net > 0), DD tail    0
intrabar        are fills decided inside a bar both ways              0 (+1 finer replay)
timeframes      does the same rule work one bar size up / down         1 per variant
markets         does the same rule work on related instruments         1 per market
spp             do random +-20% joint parameter moves keep the edge   ``spp_draws``
null            does the tape with its order destroyed do as well      up to ``null_draws``
==============  ====================================================  ==========================

The thresholds are the fixed house gates (``Thresholds``); a project does not tune them.

The ``null`` permutes bars (Masters / neurotrader style): each bar becomes (open gap, high, low,
close relative to its open); session-opening gaps are permuted among session openings and the
intraday bars' gaps and (high, low, close) shapes are permuted independently, then prices are
rebuilt by chaining from the first open. ``group="all"`` permutes across the whole tape (destroys
every temporal pattern: intraday order, daily trend, volatility clustering); ``group="session"``
permutes inside each session only (keeps each session's net move and gap: tests intraday order
alone). Timestamps and volume stay in place. The p-value is ``(1 + K) / (B + 1)`` with K = null
Sharpes >= observed; errored draws count as >= observed. It stops early once failure is certain.

It does not account for search. A pick still needs DSR at the logged N (``qik.selection``),
routed by ``g4_route``. When DSR is ambiguous (0.60-0.85) or N is uncertain, ``selector_null``
replays the whole automated selection on 19 permuted tapes (the "mini-canary").
"""

from __future__ import annotations

import math
import multiprocessing as mp
import time
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import date
from typing import Any, Literal

import numpy as np
import pandas as pd

from qik.metrics import ann_sharpe, max_drawdown_usd, session_pnl

__all__ = [
    "Check",
    "GauntletReport",
    "ParamSpec",
    "Score",
    "Subject",
    "Thresholds",
    "Variant",
    "ambiguous_exits",
    "futures_session",
    "permute_bars",
    "pooled_rank",
    "run_gauntlet",
    "score",
    "spp_configs",
    "selector_null",
    "g4_route",
    "DSR_PASS",
    "DSR_FAIL",
]

Status = Literal["PASS", "FAIL", "NOT_APPLICABLE", "NOT_RUN"]
NullGroup = Literal["all", "session"]
RunFn = Callable[[Any, Any], pd.DataFrame]


@dataclass(frozen=True)
class Thresholds:
    """House gates (Research Validation Routing, gauntlet amendment). Fixed: not a per-project knob."""

    stress_sharpe_ratio: float = 0.5
    trade_mc_draws: int = 2000
    trade_mc_block: int = 10
    trade_mc_prob_positive: float = 0.90
    intrabar_max_share: float = 0.02
    intrabar_net_ratio: float = 0.75
    variant_pass_share: float = 2 / 3
    market_pass_share: float = 0.5
    spp_draws: int = 200
    spp_rel: float = 0.20
    spp_median_sharpe_ratio: float = 0.5
    null_draws: int = 199
    null_alpha: float = 0.05


@dataclass(frozen=True)
class ParamSpec:
    """One numeric config field moved by the parameter permutation.

    ``lo`` / ``hi`` clip the draw to the field's domain. A zero-valued field moves by
    ``rel * (hi - lo)`` and needs both bounds. ``integer`` rounds the draw.
    """

    name: str
    lo: float | None = None
    hi: float | None = None
    integer: bool = False


@dataclass(frozen=True)
class Variant:
    """An alternative (config, prepared data, eligible sessions): another bar size or instrument."""

    cfg: Any
    data: Any
    eligible: Sequence[date | pd.Timestamp | str]


@dataclass(frozen=True)
class IntrabarResult:
    """``ambiguous``: trades whose exit bar could have hit stop and target; ``finer_net``: net P&L
    of a replay on finer bars, or None when the adapter has none."""

    ambiguous: int
    finer_net: float | None = None


@dataclass
class Subject:
    """Everything the gauntlet needs about one strategy package.

    ``run(cfg, data)`` returns trades with a P&L column and a session-date column (``pnl_col`` /
    ``session_col``). ``data`` is ``prepare(raw)``; ``prepare`` must not depend on ``cfg``. For a
    multi-instrument tape or a custom null, supply ``null_data(rng) -> data`` instead of relying on
    ``permute_bars(raw)``.
    """

    name: str
    run: RunFn
    cfg: Any
    raw: pd.DataFrame
    eligible: Sequence[date | pd.Timestamp | str]
    prepare: Callable[[pd.DataFrame], Any] = lambda raw: raw
    pnl_col: str = "pnl"
    session_col: str = "session_date"
    stress: Mapping[str, Any] = field(default_factory=dict)
    params: Sequence[ParamSpec] = ()
    timeframes: Mapping[str, Variant] = field(default_factory=dict)
    markets: Mapping[str, Variant] = field(default_factory=dict)
    intrabar: Callable[[pd.DataFrame, Any], IntrabarResult] | None = None
    null_group: NullGroup = "all"
    null_data: Callable[[np.random.Generator], Any] | None = None
    session_of: Callable[[pd.DatetimeIndex], np.ndarray] = field(
        default=lambda ix: futures_session(ix)
    )
    tick: float | None = None
    periods_per_year: float = 252.0


@dataclass(frozen=True)
class Score:
    trades: int
    net: float
    sharpe: float


@dataclass
class Check:
    name: str
    status: Status
    summary: str
    runs: int = 0
    seconds: float = 0.0
    stats: dict[str, Any] = field(default_factory=dict)


@dataclass
class GauntletReport:
    subject: str
    status: Literal["PASS", "FAIL", "INCOMPLETE"]
    base: Score | None
    checks: list[Check]
    seconds: float
    seed: int
    thresholds: Thresholds

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "status": self.status,
            "base": None if self.base is None else asdict(self.base),
            "seconds": round(self.seconds, 2),
            "seed": self.seed,
            "thresholds": asdict(self.thresholds),
            "checks": [asdict(c) for c in self.checks],
        }

    def to_markdown(self) -> str:
        base = (
            "n/a"
            if self.base is None
            else f"{self.base.trades} trades, net {self.base.net:,.0f}, Sharpe {self.base.sharpe:.2f}"
        )
        lines = [
            f"# Gauntlet: {self.subject}",
            "",
            f"**{self.status}** · base {base} · {self.seconds:.0f} s · seed {self.seed}",
            "",
            "| check | status | runs | s | result |",
            "|---|---|---:|---:|---|",
        ]
        lines += [
            f"| {c.name} | {c.status} | {c.runs} | {c.seconds:.1f} | {c.summary} |"
            for c in self.checks
        ]
        return "\n".join(lines) + "\n"


def futures_session(index: pd.DatetimeIndex) -> np.ndarray:
    """CME session date: New York date of ``ts + 6h`` (the 18:00 ET open belongs to the next day)."""

    ny = index.tz_convert("America/New_York") if index.tz is not None else index
    return np.asarray((ny + pd.Timedelta(hours=6)).date)


def score(
    trades: pd.DataFrame,
    eligible: Sequence[date | pd.Timestamp | str],
    *,
    pnl_col: str = "pnl",
    session_col: str = "session_date",
    periods_per_year: float = 252.0,
) -> tuple[Score, pd.Series]:
    """Net, trade count and eligible-session Sharpe of P&L (sample std); also the session P&L."""

    daily = session_pnl(trades, eligible, pnl_col=pnl_col, session_col=session_col)
    sharpe = ann_sharpe(daily.to_numpy(float), periods_per_year=periods_per_year)
    net = float(trades[pnl_col].sum()) if len(trades) else 0.0
    return Score(len(trades), net, float(sharpe) if math.isfinite(sharpe) else 0.0), daily


def _score(subject: Subject, cfg: Any, data: Any, eligible: Sequence[Any] | None = None) -> Score:
    trades = subject.run(cfg, data)
    return score(
        trades,
        subject.eligible if eligible is None else eligible,
        pnl_col=subject.pnl_col,
        session_col=subject.session_col,
        periods_per_year=subject.periods_per_year,
    )[0]


def permute_bars(
    raw: pd.DataFrame,
    rng: np.random.Generator,
    *,
    group: NullGroup = "all",
    session_of: Callable[[pd.DatetimeIndex], np.ndarray] = futures_session,
    tick: float | None = None,
) -> pd.DataFrame:
    """Bar permutation of an OHLC(V) tape; index, volume and other columns stay in place.

    See the module docstring for what each ``group`` keeps. Prices are rebuilt from the first open;
    ``tick`` rounds them and then re-imposes low <= open, close <= high.
    """

    cols = ["open", "high", "low", "close"]
    o, h, lo, c = (raw[k].to_numpy(float) for k in cols)
    if len(o) < 2:
        return raw.copy()
    if (np.minimum.reduce([o, h, lo, c]) <= 0).any() or not np.isfinite([o, h, lo, c]).all():
        raise ValueError("permute_bars needs positive finite OHLC")
    prev_c = np.concatenate([[o[0]], c[:-1]])
    gap = np.log(o / prev_c)
    shape = np.column_stack([np.log(h / o), np.log(lo / o), np.log(c / o)])
    sessions = session_of(pd.DatetimeIndex(raw.index))
    first = np.ones(len(o), bool)
    first[1:] = sessions[1:] != sessions[:-1]
    first[0] = True

    new_gap = gap.copy()
    new_shape = shape.copy()
    if group == "all":
        opens = np.flatnonzero(first)
        inner = np.flatnonzero(~first)
        new_gap[opens[1:]] = gap[rng.permutation(opens[1:])]
        new_gap[inner] = gap[rng.permutation(inner)]
        new_shape = shape[rng.permutation(len(o))]
    elif group == "session":
        starts = np.flatnonzero(first)
        ends = np.append(starts[1:], len(o))
        for s, e in zip(starts, ends, strict=True):
            if e - s > 1:
                new_gap[s + 1 : e] = gap[s + 1 + rng.permutation(e - s - 1)]
            new_shape[s:e] = shape[s + rng.permutation(e - s)]
    else:
        raise ValueError(f"unknown group: {group!r}")
    new_gap[0] = 0.0

    log_o = np.empty(len(o))
    log_o[0] = math.log(o[0])
    step = new_gap + new_shape[:, 2]
    log_c = log_o[0] + np.cumsum(step)
    log_o[1:] = log_c[:-1] + new_gap[1:]
    out = raw.copy()
    no = np.exp(log_o)
    nh = np.exp(log_o + new_shape[:, 0])
    nl = np.exp(log_o + new_shape[:, 1])
    nc = np.exp(log_c)
    if tick:
        no, nh, nl, nc = (np.round(x / tick) * tick for x in (no, nh, nl, nc))
    nh = np.maximum.reduce([nh, no, nc, nl])
    nl = np.minimum.reduce([nl, no, nc, nh])
    out["open"], out["high"], out["low"], out["close"] = no, nh, nl, nc
    return out


def spp_configs(
    cfg: Any, params: Sequence[ParamSpec], n: int, rng: np.random.Generator, rel: float
) -> list[Any]:
    """``n`` configs with every ``params`` field moved jointly, uniform in ``[-rel, +rel]``."""

    names = {f.name for f in fields(cfg)}
    for p in params:
        if p.name not in names:
            raise ValueError(f"{p.name!r} is not a field of {type(cfg).__name__}")
    out: list[Any] = []
    for _ in range(n):
        changes: dict[str, Any] = {}
        for p in params:
            v = float(getattr(cfg, p.name))
            if v == 0.0:
                if p.lo is None or p.hi is None:
                    raise ValueError(f"{p.name!r} is 0: give lo and hi to set its step")
                x = rng.uniform(-rel, rel) * (p.hi - p.lo)
            else:
                x = v * (1.0 + rng.uniform(-rel, rel))
            if p.lo is not None:
                x = max(x, p.lo)
            if p.hi is not None:
                x = min(x, p.hi)
            changes[p.name] = round(x) if p.integer else x
        out.append(replace(cfg, **changes))
    return out


def ambiguous_exits(
    high: np.ndarray, low: np.ndarray, exit_i: np.ndarray, stop: np.ndarray, target: np.ndarray
) -> np.ndarray:
    """Per trade: did its exit bar's range reach both the stop and the target (order unknowable)?"""

    hi, lo = high[exit_i], low[exit_i]
    top, bottom = np.maximum(stop, target), np.minimum(stop, target)
    return (hi >= top) & (lo <= bottom) & np.isfinite(stop) & np.isfinite(target)


def pooled_rank(scores: pd.DataFrame) -> pd.DataFrame:
    """Rank configs (rows) on several markets (columns, higher = better) for one pooled pick.

    Returns ``median_rank`` (1 = best) and ``worst_rank`` per config, sorted best first. Pick on the
    median; a config that is bad anywhere shows in ``worst_rank``.
    """

    ranks = scores.rank(axis=0, ascending=False, method="average")
    out = pd.DataFrame({"median_rank": ranks.median(axis=1), "worst_rank": ranks.max(axis=1)})
    return out.sort_values(["median_rank", "worst_rank"])


def _block_bootstrap(
    daily: np.ndarray, rng: np.random.Generator, draws: int, block: int
) -> np.ndarray:
    n = len(daily)
    k = math.ceil(n / block)
    starts = rng.integers(0, max(n - block + 1, 1), size=(draws, k))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(draws, -1)[:, :n]
    return daily[np.minimum(idx, n - 1)]


# --- parallel workers (fork: children inherit the subject; nothing is pickled but seeds/configs) ---

_WORKER: dict[str, Any] = {}


def _null_draw(seed: tuple[int, int]) -> float | str:
    subject: Subject = _WORKER["subject"]
    rng = np.random.default_rng(list(seed))
    try:
        if subject.null_data is not None:
            data = subject.null_data(rng)
        else:
            tape = permute_bars(
                subject.raw,
                rng,
                group=subject.null_group,
                session_of=subject.session_of,
                tick=subject.tick,
            )
            data = subject.prepare(tape)
        return _score(subject, subject.cfg, data).sharpe
    except Exception as exc:  # noqa: BLE001 - an errored draw is counted against the strategy
        return f"{type(exc).__name__}: {exc}"[:300]


def _spp_draw(cfg: Any) -> tuple[float, float] | str:
    subject: Subject = _WORKER["subject"]
    try:
        s = _score(subject, cfg, _WORKER["data"])
        return s.net, s.sharpe
    except Exception as exc:  # noqa: BLE001 - an errored draw is counted against the strategy
        return f"{type(exc).__name__}: {exc}"[:300]


def _map(fn: Callable[[Any], Any], items: Sequence[Any], jobs: int) -> list[Any]:
    if jobs <= 1 or len(items) <= 1:
        return [fn(x) for x in items]
    ctx = mp.get_context("fork")
    with ProcessPoolExecutor(max_workers=jobs, mp_context=ctx) as pool:
        return list(pool.map(fn, items, chunksize=max(1, len(items) // (jobs * 4))))


# --- checks ---


def _check_stress(subject: Subject, data: Any, base: Score, t: Thresholds) -> Check:
    if not subject.stress:
        return Check("stress", "NOT_APPLICABLE", "no stress variants supplied")
    rows: dict[str, dict[str, float]] = {}
    ok = True
    for name, cfg in subject.stress.items():
        s = _score(subject, cfg, data)
        passed = s.net > 0 and s.sharpe >= t.stress_sharpe_ratio * base.sharpe
        ok &= passed
        rows[name] = {"net": round(s.net, 2), "sharpe": round(s.sharpe, 3), "pass": passed}
    text = "; ".join(f"{k} Sh {v['sharpe']:.2f}" for k, v in rows.items())
    return Check("stress", "PASS" if ok else "FAIL", text, runs=len(rows), stats=rows)


def _check_trade_mc(daily: pd.Series, rng: np.random.Generator, t: Thresholds) -> Check:
    x = daily.to_numpy(float)
    if len(x) < 2 * t.trade_mc_block:
        return Check("trade_mc", "NOT_APPLICABLE", "too few sessions")
    sims = _block_bootstrap(x, rng, t.trade_mc_draws, t.trade_mc_block)
    nets = sims.sum(axis=1)
    dds = np.array([max_drawdown_usd(row) for row in sims])
    p_pos = float((nets > 0).mean())
    stats = {
        "p_net_positive": round(p_pos, 4),
        "net_p5": round(float(np.percentile(nets, 5)), 2),
        "max_dd_actual": round(max_drawdown_usd(x), 2),
        "max_dd_p50": round(float(np.percentile(dds, 50)), 2),
        "max_dd_p5": round(float(np.percentile(dds, 5)), 2),
    }
    status: Status = "PASS" if p_pos >= t.trade_mc_prob_positive else "FAIL"
    text = f"P(net>0) {p_pos:.2f}; max DD actual {stats['max_dd_actual']:,.0f}, 5% worst {stats['max_dd_p5']:,.0f}"
    return Check("trade_mc", status, text, stats=stats)


def _check_intrabar(
    subject: Subject, trades: pd.DataFrame, data: Any, base: Score, t: Thresholds
) -> Check:
    if subject.intrabar is None:
        return Check("intrabar", "NOT_APPLICABLE", "no stop+target in the same bar (adapter)")
    r = subject.intrabar(trades, data)
    share = r.ambiguous / max(len(trades), 1)
    stats: dict[str, Any] = {
        "ambiguous": r.ambiguous,
        "share": round(share, 4),
        "finer_net": r.finer_net,
    }
    if share <= t.intrabar_max_share:
        return Check("intrabar", "PASS", f"{share:.1%} ambiguous exits", stats=stats)
    if r.finer_net is None:
        return Check(
            "intrabar", "FAIL", f"{share:.1%} ambiguous exits and no finer replay", stats=stats
        )
    ok = r.finer_net >= t.intrabar_net_ratio * base.net
    return Check(
        "intrabar",
        "PASS" if ok else "FAIL",
        f"{share:.1%} ambiguous; finer replay net {r.finer_net:,.0f} vs {base.net:,.0f}",
        runs=1,
        stats=stats,
    )


def _check_variants(
    name: str, subject: Subject, variants: Mapping[str, Variant], need: float
) -> Check:
    if not variants:
        return Check(name, "NOT_APPLICABLE", "none supplied")
    rows: dict[str, dict[str, Any]] = {}
    for key, v in variants.items():
        s = _score(subject, v.cfg, v.data, v.eligible)
        rows[key] = {
            "trades": s.trades,
            "net": round(s.net, 2),
            "sharpe": round(s.sharpe, 3),
            "pass": s.net > 0 and s.sharpe > 0,
        }
    share = sum(r["pass"] for r in rows.values()) / len(rows)
    text = "; ".join(f"{k} Sh {r['sharpe']:.2f}" for k, r in rows.items())
    return Check(
        name,
        "PASS" if share >= need else "FAIL",
        f"{share:.0%} positive: {text}",
        runs=len(rows),
        stats=rows,
    )


def _check_spp(
    subject: Subject, data: Any, base: Score, t: Thresholds, seed: int, jobs: int
) -> Check:
    if not subject.params:
        return Check("spp", "NOT_APPLICABLE", "no numeric parameters declared")
    rng = np.random.default_rng([seed, 1])
    cfgs = spp_configs(subject.cfg, subject.params, t.spp_draws, rng, t.spp_rel)
    _WORKER.update(subject=subject, data=data)
    results = _map(_spp_draw, cfgs, jobs)
    ok_rows = [r for r in results if not isinstance(r, str)]
    error_text = sorted({r for r in results if isinstance(r, str)})
    errors = len(results) - len(ok_rows)
    nets = np.array([r[0] for r in ok_rows] + [-math.inf] * errors)
    shs = np.array([r[1] for r in ok_rows] + [-math.inf] * errors)
    med_sh, med_net = float(np.median(shs)), float(np.median(nets))
    stats = {
        "draws": len(results),
        "errors": errors,
        "error_examples": error_text[:3],
        "sharpe_p10": float(np.percentile(shs, 10)),
        "sharpe_p50": med_sh,
        "sharpe_p90": float(np.percentile(shs, 90)),
        "net_p50": med_net,
        "share_profitable": float((nets > 0).mean()),
        "params": [p.name for p in subject.params],
    }
    ok = med_net > 0 and med_sh >= t.spp_median_sharpe_ratio * base.sharpe
    text = (
        f"median Sh {med_sh:.2f} vs base {base.sharpe:.2f} (p10 {stats['sharpe_p10']:.2f}); "
        f"{stats['share_profitable']:.0%} profitable"
    )
    return Check("spp", "PASS" if ok else "FAIL", text, runs=len(results), stats=stats)


def _check_null(subject: Subject, base: Score, t: Thresholds, seed: int, jobs: int) -> Check:
    b = t.null_draws
    k_fail = math.floor(t.null_alpha * (b + 1))  # (1 + K) / (B + 1) > alpha  <=>  K >= k_fail
    _WORKER.update(subject=subject)
    batch = max(jobs, 1) * 4
    bar = base.sharpe - 1e-9 * max(1.0, abs(base.sharpe))  # float-level ties count against
    k = errors = done = 0
    sharpes: list[float] = []
    error_text: list[str] = []
    early = False
    for start in range(0, b, batch):
        seeds = [(seed, 1000 + i) for i in range(start, min(start + batch, b))]
        for r in _map(_null_draw, seeds, jobs):
            done += 1
            if isinstance(r, str):
                errors += 1
                k += 1
                if len(error_text) < 3 and r not in error_text:
                    error_text.append(r)
            else:
                sharpes.append(r)
                k += r >= bar
        if k >= k_fail:
            early = done < b
            break
    p = (1 + k) / (b + 1)
    stats: dict[str, Any] = {
        "group": "custom" if subject.null_data is not None else subject.null_group,
        "B": b,
        "completed": done,
        "K": k,
        "errors": errors,
        "error_examples": error_text,
        "early_failure": early,
        "null_sharpe_p50": float(np.median(sharpes)) if sharpes else None,
        "null_sharpe_p95": float(np.percentile(sharpes, 95)) if sharpes else None,
    }
    if early:
        lo = (1 + k) / (b + 1)
        stats["p_interval"] = [lo, min(1.0, (1 + k + b - done) / (b + 1))]
        return Check(
            "null",
            "FAIL",
            f"K={k} after {done}/{b}: p >= {lo:.3f}, failure certain"
            + (f"; {errors} draws ERRORED: {error_text[0]}" if errors else ""),
            runs=done,
            stats=stats,
        )
    stats["p"] = p
    ok = p <= t.null_alpha
    med = stats["null_sharpe_p50"]
    text = f"p = {p:.3f} (K={k}/{b}); null median Sh " + ("n/a" if med is None else f"{med:.2f}")
    if errors:
        text += f"; {errors} draws ERRORED (counted against): {error_text[0]}"
    return Check("null", "PASS" if ok else "FAIL", text, runs=done, stats=stats)


# --- mini-canary: the whole automated selector replayed on permuted tapes ---

DSR_PASS = 0.85  # G4: DSR at the logged N at or above this passes without a selector null
DSR_FAIL = 0.60  # below this G4 fails; in between (or N uncertain) run ``selector_null``


def g4_route(
    dsr: float | None, *, n_certain: bool = True
) -> Literal["PASS", "FAIL", "SELECTOR_NULL"]:
    """G4 routing (routing § 0): DSR >= 0.85 PASS, < 0.60 FAIL, otherwise run ``selector_null``.

    ``dsr=None`` (undefined) or ``n_certain=False`` (adaptive search, fuzzy trial count) always
    routes to the selector null, unless DSR is already below the failure line.
    """

    if dsr is not None and dsr < DSR_FAIL:
        return "FAIL"
    if dsr is None or not n_certain or dsr < DSR_PASS:
        return "SELECTOR_NULL"
    return "PASS"


def _selector_draw(seed: tuple[int, int]) -> float | str:
    w = _WORKER["selector"]
    rng = np.random.default_rng(list(seed))
    try:
        if w["null_data"] is not None:
            data = w["null_data"](rng)
        else:
            tape = permute_bars(
                w["raw"], rng, group=w["group"], session_of=w["session_of"], tick=w["tick"]
            )
            data = w["prepare"](tape)
        return float(w["select"](data))
    except Exception as exc:  # noqa: BLE001 - an errored draw is counted against the strategy
        return f"{type(exc).__name__}: {exc}"[:300]


def selector_null(
    raw: pd.DataFrame,
    select: Callable[[Any], float],
    *,
    prepare: Callable[[pd.DataFrame], Any] = lambda raw: raw,
    observed: float | None = None,
    tapes: int = 19,
    seed: int = 0,
    group: NullGroup = "all",
    session_of: Callable[[pd.DatetimeIndex], np.ndarray] = futures_session,
    tick: float | None = None,
    null_data: Callable[[np.random.Generator], Any] | None = None,
    jobs: int = 1,
) -> Check:
    """Mini-canary: replay the whole automated selection on ``tapes`` bar-permuted tapes.

    ``select(data)`` must run the complete deterministic search + pick on ``data`` and return the
    claimed statistic, e.g. the picked config's Sharpe gain over H0 on that same tape, or the
    picked Sharpe. ``observed`` defaults to ``select(prepare(raw))``. Exact Monte-Carlo test with
    p = (1 + K) / (tapes + 1), K = null statistics >= observed (errors count as >= observed). With
    the default 19 tapes, PASS needs K = 0 (p = 0.05), so the run stops at the first tape that
    ties or beats the real result. A failing selector usually costs one to three tapes.
    """

    t0 = time.perf_counter()
    if observed is None:
        observed = float(select(prepare(raw)))
    _WORKER["selector"] = {
        "raw": raw,
        "select": select,
        "prepare": prepare,
        "group": group,
        "session_of": session_of,
        "tick": tick,
        "null_data": null_data,
    }
    bar = observed - 1e-9 * max(1.0, abs(observed))
    nulls: list[float] = []
    errors: list[str] = []
    done = 0
    try:
        for start in range(0, tapes, max(jobs, 1)):
            seeds = [(seed, 5000 + i) for i in range(start, min(start + max(jobs, 1), tapes))]
            for r in _map(_selector_draw, seeds, jobs):
                done += 1
                if isinstance(r, str):
                    errors.append(r)
                else:
                    nulls.append(r)
            if errors or any(x >= bar for x in nulls):
                break
    finally:
        _WORKER.pop("selector", None)
    k = len(errors) + sum(x >= bar for x in nulls)
    stats: dict[str, Any] = {
        "observed": observed,
        "tapes": tapes,
        "completed": done,
        "K": k,
        "errors": len(errors),
        "error_examples": sorted(set(errors))[:3],
        "null_max": max(nulls) if nulls else None,
        "group": "custom" if null_data is not None else group,
    }
    if k:
        text = f"a permuted tape matched the real result after {done}/{tapes} tapes (p >= {(1 + k) / (tapes + 1):.2f})"
        if errors:
            text += f"; {len(errors)} ERRORED: {errors[0]}"
        return Check(
            "selector_null", "FAIL", text, runs=done, seconds=time.perf_counter() - t0, stats=stats
        )
    p = 1 / (tapes + 1)
    stats["p"] = p
    text = f"real {observed:.3f} beat all {tapes} permuted tapes (max {stats['null_max']:.3f}); p = {p:.3f}"
    return Check(
        "selector_null", "PASS", text, runs=done, seconds=time.perf_counter() - t0, stats=stats
    )


def run_gauntlet(
    subject: Subject,
    *,
    thresholds: Thresholds | None = None,
    seed: int = 0,
    jobs: int = 1,
    fail_fast: bool = True,
    skip: Sequence[str] = (),
    log: Callable[[str], None] | None = None,
) -> GauntletReport:
    """Run the checks cheapest first. ``skip`` names checks to leave NOT_RUN (reported, never PASS)."""

    t = thresholds or Thresholds()
    t0 = time.perf_counter()
    data = subject.prepare(subject.raw)
    trades = subject.run(subject.cfg, data)
    base, daily = score(
        trades,
        subject.eligible,
        pnl_col=subject.pnl_col,
        session_col=subject.session_col,
        periods_per_year=subject.periods_per_year,
    )
    checks: list[Check] = []
    base_ok = base.net > 0 and base.sharpe > 0
    checks.append(
        Check(
            "base",
            "PASS" if base_ok else "FAIL",
            f"{base.trades} trades, net {base.net:,.0f}, Sharpe {base.sharpe:.2f}",
            runs=1,
            seconds=time.perf_counter() - t0,
        )
    )
    steps: list[tuple[str, Callable[[], Check]]] = [
        ("stress", lambda: _check_stress(subject, data, base, t)),
        ("trade_mc", lambda: _check_trade_mc(daily, np.random.default_rng([seed, 2]), t)),
        ("intrabar", lambda: _check_intrabar(subject, trades, data, base, t)),
        (
            "timeframes",
            lambda: _check_variants(
                "timeframes", subject, subject.timeframes, t.variant_pass_share
            ),
        ),
        (
            "markets",
            lambda: _check_variants("markets", subject, subject.markets, t.market_pass_share),
        ),
        ("spp", lambda: _check_spp(subject, data, base, t, seed, jobs)),
        ("null", lambda: _check_null(subject, base, t, seed, jobs)),
    ]
    stopped = not base_ok and fail_fast
    for name, fn in steps:
        if stopped or name in skip:
            reason = "skipped by request" if name in skip else "not reached (fail-fast)"
            checks.append(Check(name, "NOT_RUN", reason))
            continue
        c0 = time.perf_counter()
        check = fn()
        check.seconds = time.perf_counter() - c0
        checks.append(check)
        if log is not None:
            log(
                f"{subject.name} {check.name}: {check.status} {check.summary} ({check.seconds:.1f}s)"
            )
        stopped = fail_fast and check.status == "FAIL"
    statuses = {c.status for c in checks}
    status: Literal["PASS", "FAIL", "INCOMPLETE"] = (
        "FAIL" if "FAIL" in statuses else "INCOMPLETE" if "NOT_RUN" in statuses else "PASS"
    )
    _WORKER.clear()
    return GauntletReport(subject.name, status, base, checks, time.perf_counter() - t0, seed, t)
