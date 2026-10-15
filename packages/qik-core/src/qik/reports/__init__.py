"""Shared research reporting helpers (QuantStats tearsheets, strategy reports)."""

from qik.reports.quantstats_report import (
    equity_curve_to_daily_returns,
    generate_quantstats_html,
    summary_stats,
    trades_to_daily_returns,
    write_report_index,
)
from qik.reports.strategy_report import format_summary_console, write_strategy_report

__all__ = [
    "equity_curve_to_daily_returns",
    "format_summary_console",
    "generate_quantstats_html",
    "summary_stats",
    "trades_to_daily_returns",
    "write_report_index",
    "write_strategy_report",
]
