"""Unified strategy report artifacts (summary JSON + returns + QuantStats HTML)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from qik.reports.quantstats_report import generate_quantstats_html, summary_stats


def format_summary_console(summary: dict[str, Any], *, prefix: str = "  ") -> str:
    """One canonical console block for headline metrics."""

    def _fmt(key: str, default: str = "n/a") -> str:
        val = summary.get(key)
        if val is None:
            return default
        if isinstance(val, float):
            if val != val:  # NaN
                return default
            if key in ("win_rate", "max_drawdown_pct", "pct_sessions_traded"):
                if key == "win_rate" and abs(val) <= 1.0:
                    return f"{val:.1%}"
                if key == "max_drawdown_pct" and abs(val) <= 1.5:
                    return f"{val:.1%}"
                return f"{val:.2f}"
            if key.endswith("_usd") or key in (
                "net_pnl_usd",
                "max_drawdown_usd",
                "avg_win",
                "avg_loss",
            ):
                return f"${val:,.0f}"
            if key in ("ann_vol",) or (key == "cagr" and abs(val) < 5):
                return f"{val:.1%}"
            if (
                "sharpe" in key
                or "sortino" in key
                or key in ("profit_factor", "skew", "lower_tail_ratio", "upper_tail_ratio")
            ):
                return f"{val:.2f}"
            if key.endswith("_pct") and "cagr" in key:
                return f"{val:.2f}"
            return f"{val:.4g}"
        return str(val)

    parts = [
        f"trades={summary.get('n_trades', 0)}",
        f"net={_fmt('net_pnl_usd')}",
        f"PF={_fmt('profit_factor')}",
        f"Sharpe={_fmt('sharpe_ann')}",
    ]
    if summary.get("n_eligible_sessions"):
        parts.append(f"Sharpe_session={_fmt('sharpe_session_ann')}")
        parts.append(f"Sharpe_active={_fmt('sharpe_active_day_ann')}")
    if summary.get("ann_vol") == summary.get("ann_vol"):
        parts.append(f"vol={_fmt('ann_vol')}")
    if summary.get("max_drawdown_usd") is not None:
        parts.append(f"maxDD={_fmt('max_drawdown_usd')}")
    n_el = summary.get("n_eligible_sessions")
    n_act = summary.get("n_active_sessions")
    if n_el:
        parts.append(f"sessions={n_act}/{n_el}")
    line = " ".join(parts)
    shape_parts = []
    if summary.get("skew") == summary.get("skew"):
        shape_parts.append(f"skew={_fmt('skew')}")
    if summary.get("lower_tail_ratio") == summary.get("lower_tail_ratio"):
        shape_parts.append(f"lower_tail={_fmt('lower_tail_ratio')}")
    if summary.get("upper_tail_ratio") == summary.get("upper_tail_ratio"):
        shape_parts.append(f"upper_tail={_fmt('upper_tail_ratio')}")
    extra = ""
    if shape_parts:
        extra += f"\n{prefix}" + " ".join(shape_parts)
    if summary.get("sharpe_definition"):
        extra += f"\n{prefix}sharpe: {summary['sharpe_definition']}"
    return f"{prefix}{line}{extra}"


def write_strategy_report(
    summary: dict[str, Any],
    returns: pd.Series,
    out_dir: Path | str,
    *,
    title: str,
    tag: str = "strategy",
    trades: pd.DataFrame | None = None,
    meta: dict[str, Any] | None = None,
    rf: float = 0.0,
    periods_per_year: int = 252,
    write_html: bool = True,
    write_returns_parquet: bool = True,
    print_console: bool = True,
) -> dict[str, Path]:
    """Write summary.json + optional trades/returns + QuantStats HTML.

    Returns a dict of artifact paths keyed by kind
    (``summary``, ``trades``, ``returns_csv``, ``returns_parquet``, ``html``).
    """

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    payload = dict(summary)
    if meta:
        payload.update(meta)

    paths: dict[str, Path] = {}
    summary_path = out / f"summary_{tag}.json"
    summary_path.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8")
    paths["summary"] = summary_path

    if trades is not None and not trades.empty:
        trades_path = out / f"trades_{tag}.csv"
        trades.to_csv(trades_path, index=False)
        paths["trades"] = trades_path

    if returns is not None and not returns.empty:
        rets = returns.copy()
        rets.name = rets.name or "Strategy"
        csv_path = out / f"returns_{tag}.csv"
        rets.to_csv(csv_path, header=True)
        paths["returns_csv"] = csv_path
        if write_returns_parquet:
            try:
                pq_path = out / f"returns_{tag}.parquet"
                rets.to_frame(name="return").to_parquet(pq_path)
                paths["returns_parquet"] = pq_path
            except ImportError:  # parquet engine (pyarrow/fastparquet) is optional
                pass

    if write_html and returns is not None and not returns.empty:
        html_path = out / f"quantstats_{tag}.html"
        generate_quantstats_html(
            returns,
            html_path,
            title=title,
            benchmark=None,
            rf=rf,
            periods_per_year=periods_per_year,
        )
        paths["html"] = html_path

    if print_console:
        print(format_summary_console(payload))
        for kind, path in paths.items():
            print(f"  {kind} → {path}")
        try:
            qs = summary_stats(returns, trades)
            print("=== QuantStats summary ===")
            for k, v in qs.items():
                print(f"  {k}: {v}")
            if "sharpe_session_ann" in payload:
                print(
                    f"  blotter sharpe_session_ann: {payload.get('sharpe_session_ann')} | "
                    f"sharpe_active_day_ann: {payload.get('sharpe_active_day_ann')}"
                )
        except Exception as exc:  # noqa: BLE001
            print(f"(summary_stats skipped: {exc})")

    return paths


__all__ = ["format_summary_console", "write_strategy_report"]
