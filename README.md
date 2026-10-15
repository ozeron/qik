# QIK

Quant Infrastructure Kit (QIK): small, typed building blocks for systematic trading research.
Plain pandas/numpy, no broker or engine dependency.

## Packages

| Package | Import | What |
|---|---|---|
| `qik-core` | `qik.indicators` | SMA/EMA/RMA/WMA/HMA, true range, ATR, Bollinger, RSI, Connors RSI, session VWAP, opening range — every convention difference (seeding, Wilder vs SMA smoothing, warm-up, ddof, …) is an explicit parameter |
| | `qik.metrics` | returns grids, Sharpe (incl. eligible-session Sharpe for discrete intraday blotters), Sortino, CAGR, drawdown, profit factor, win rate, expectancy, trade stats |
| | `qik.blotter` | closed-trade blotter contract: column names, units, normalisation |
| | `qik.runs` | run records: one directory per backtest run (`meta.json` with git sha + params hash, `metrics.json`, `trades.parquet`) |
| | `qik.reports` | QuantStats tearsheets and console summaries (`pip install qik-core[reports]`) |
| `qik-synthetic` | `qik.synthetic` | synthetic OHLCV generation, calibration and diagnostics |

`qik` is a namespace package: each distribution adds its own subpackages.

## Development

```bash
uv sync --all-packages
uv run pytest packages/*/tests
```

Packages live under `packages/` as `uv` workspace members.

This repository is published one-way from a private monorepo; open issues or PRs here and they
are carried over.
