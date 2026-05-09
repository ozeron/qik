"""Primary generation engine for class-based synthetic workflows."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from hashlib import blake2b
from typing import Iterator

import numpy as np

from .context import DefaultMarketContextProvider
from .errors import SyntheticContextError, SyntheticValidationError
from .horizon import bars_from_span, generate_timestamps, parse_interval
from .protocols import MarketContextProvider, PathModel
from .registry import resolve_model
from .types import (
    SyntheticBatch,
    SyntheticChunk,
    SyntheticContext,
    SyntheticMetadata,
    SyntheticRequest,
)


_INTERNAL_BLOCK_SIZE = 200_000


@dataclass(frozen=True)
class _RunPlan:
    request: SyntheticRequest
    context: SyntheticContext
    n_bars: int
    params: dict[str, object]
    run_id: str
    interval_step: object


@dataclass(frozen=True)
class _OhlcvBlock:
    block_index: int
    block_start: int
    timestamps: tuple[datetime, ...]
    ohlcv: dict[str, object]


def _seed_to_int(seed_data: bytes) -> int:
    return int.from_bytes(blake2b(seed_data, digest_size=8).digest(), "big", signed=False)


def _stable_id(*parts: object) -> str:
    return blake2b(repr(tuple(_stable_param_repr(part) for part in parts)).encode("utf-8"), digest_size=8).hexdigest()


@dataclass
class SyntheticEngine:
    """Orchestrate request validation, generation, and payload conversion."""

    model: str | PathModel
    context_provider: MarketContextProvider | None = None
    seed: int | None = None
    _context_cache: dict[tuple[str | None, str | None, str | None, str], SyntheticContext] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        if isinstance(self.model, str):
            self.model = resolve_model(self.model)
        if self.context_provider is None:
            self.context_provider = DefaultMarketContextProvider()

    def fit(self, history: Sequence[Mapping[str, object]] | None = None) -> "SyntheticEngine":
        """Fit the model on historical rows before generation."""
        self.model.fit(history, context=None)
        return self

    def fit_frame(self, frame) -> "SyntheticEngine":
        """Fit the model from an OHLCV dataframe when supported."""
        self.model.fit(frame, context=None)
        return self

    def fit_profile(self, profile) -> "SyntheticEngine":
        """Fit the model from an OHLCV calibration profile when supported."""
        self.model.fit(profile, context=None)
        return self

    def generate(self, request: SyntheticRequest) -> SyntheticBatch:
        """Materialize a full synthetic batch for one request."""
        plan = self._prepare_run(request)
        bars = tuple(
            {
                "timestamp": block.timestamps[idx],
                "open": float(block.ohlcv["open"][idx]),
                "high": float(block.ohlcv["high"][idx]),
                "low": float(block.ohlcv["low"][idx]),
                "close": float(block.ohlcv["close"][idx]),
                "volume": float(block.ohlcv["volume"][idx]),
            }
            for block in self._iter_ohlcv_blocks(plan)
            for idx in range(len(block.timestamps))
        )
        return SyntheticBatch(
            bars=bars,
            context=plan.context,
            request=plan.request,
            metadata=self._build_metadata(plan.request, plan.context),
        )

    def iter_generate(self, request: SyntheticRequest, *, chunk_size: int = 200_000) -> Iterator[SyntheticChunk]:
        """Yield generated chunks to support large horizons safely.

        Chunk boundaries are deterministic and resume-safe because each chunk uses a
        seed derived from (request run seed, model, and chunk index).
        """
        if chunk_size <= 0:
            raise SyntheticValidationError("chunk_size must be greater than 0")

        plan = self._prepare_run(request)

        if plan.n_bars == 0:
            return
        chunk_index = 0

        for block in self._iter_ohlcv_blocks(plan):
            block_n = len(block.timestamps)
            for offset_start in range(0, block_n, chunk_size):
                chunk_n = min(chunk_size, block_n - offset_start)
                rows = tuple(
                    {
                        "timestamp": block.timestamps[offset_start + offset],
                        "open": float(block.ohlcv["open"][offset_start + offset]),
                        "high": float(block.ohlcv["high"][offset_start + offset]),
                        "low": float(block.ohlcv["low"][offset_start + offset]),
                        "close": float(block.ohlcv["close"][offset_start + offset]),
                        "volume": float(block.ohlcv["volume"][offset_start + offset]),
                    }
                    for offset in range(chunk_n)
                )
                yield SyntheticChunk(bars=rows, chunk_index=chunk_index, run_id=plan.run_id)
                chunk_index += 1

    def iter_generate_frames(self, request: SyntheticRequest, *, chunk_size: int = 200_000, compact_identity: bool = False):
        """Yield generated candle frames without materializing Python row dicts."""
        if chunk_size <= 0:
            raise SyntheticValidationError("chunk_size must be greater than 0")

        plan = self._prepare_run(request)
        if plan.n_bars == 0:
            return

        for block in self._iter_ohlcv_blocks(plan):
            block_n = len(block.timestamps)
            for offset_start in range(0, block_n, chunk_size):
                chunk_n = min(chunk_size, block_n - offset_start)
                yield self._frame_from_block(
                    plan,
                    block,
                    offset_start=offset_start,
                    chunk_n=chunk_n,
                    compact_identity=compact_identity,
                )

    def generate_frame(self, request: SyntheticRequest, *, compact_identity: bool = False):
        """Materialize a generated candle frame via the columnar fast path."""
        plan = self._prepare_run(request)
        return self._generate_frame_from_plan(plan, compact_identity=compact_identity)

    def generate_backtesting_payload(
        self,
        request: SyntheticRequest,
        *,
        run_context: Mapping[str, str] | None = None,
        compact_identity: bool = False,
    ) -> Mapping[str, object]:
        """Generate a backtesting payload without building `SyntheticBatch.bars`."""
        from .backtesting import to_backtesting_payload_from_frame

        plan = self._prepare_run(request)
        frame = self._generate_frame_from_plan(plan, compact_identity=compact_identity)
        return to_backtesting_payload_from_frame(
            frame,
            metadata=self._build_metadata(plan.request, plan.context),
            identity=self._identity_for_plan(plan),
            n_bars=plan.n_bars,
            run_context=run_context,
        )

    def iter_generate_many_frames(
        self,
        requests: Sequence[SyntheticRequest],
        *,
        chunk_size: int = 200_000,
        compact_identity: bool = False,
    ):
        """Yield `(request_index, frame)` chunks for many requests sequentially."""
        for index, request in enumerate(requests):
            for frame in self.iter_generate_frames(
                request,
                chunk_size=chunk_size,
                compact_identity=compact_identity,
            ):
                yield index, frame

    def generate_many(
        self,
        requests: Sequence[SyntheticRequest],
        *,
        strategy: str = "materialize",
    ) -> Sequence[SyntheticBatch]:
        """Batch-generation helper with request grouping.

        For very large workloads, prefer `chunk_size` in `iter_generate` so memory grows
        with chunk width instead of total generated bars.
        """
        if strategy not in {"materialize", "iter"}:
            raise SyntheticValidationError("strategy must be materialize or iter")

        normalized = [self._normalize_request(req) for req in requests]
        grouped: dict[tuple[str, str, str | None, str | None, str | None, str], list[tuple[int, SyntheticRequest]]] = {}
        metadata_cache: dict[int, tuple[SyntheticContext, int]] = {}
        for idx, req in enumerate(normalized):
            context, n_bars = self._resolve_run_context(req)
            key = (
                req.model,
                context.asset_class,
                context.interval,
                context.trading_hours,
                context.symbol,
                req.interval,
            )
            grouped.setdefault(key, []).append((idx, req))
            metadata_cache[idx] = (context, n_bars)

        out: list[SyntheticBatch | None] = [None] * len(requests)
        for members in grouped.values():
            for idx, req in members:
                context, _ = metadata_cache[idx]
                self._validate_context(context, req)
                if strategy == "materialize":
                    out[idx] = self.generate(req)
                else:
                    rows = tuple(bar for chunk in self.iter_generate(req) for bar in chunk.bars)
                    out[idx] = SyntheticBatch(
                        bars=rows,
                        context=context,
                        request=req,
                        metadata=self._build_metadata(req, context),
                    )

        return tuple(out)

    def _prepare_run(self, request: SyntheticRequest) -> _RunPlan:
        request = self._normalize_request(request)
        context, n_bars = self._resolve_run_context(request)
        self._validate_context(context, request)
        params = dict(request.model_params)
        params.setdefault("trading_hours", context.trading_hours)
        params.setdefault("asset_class", context.asset_class)
        params["start_price"] = float(request.start_price)
        return _RunPlan(
            request=request,
            context=context,
            n_bars=n_bars,
            params=params,
            run_id=self._run_id(request),
            interval_step=parse_interval(context.interval),
        )

    def _iter_ohlcv_blocks(self, plan: _RunPlan):
        timestamp_cursor = plan.request.start_ts
        last_close = float(plan.request.start_price)
        for block_index, block_start in enumerate(range(0, plan.n_bars, _INTERNAL_BLOCK_SIZE)):
            block_n = min(_INTERNAL_BLOCK_SIZE, plan.n_bars - block_start)
            params = dict(plan.params)
            params["start_price"] = last_close
            ohlcv = self.model.simulate_batch(
                plan.context,
                block_n,
                seed=self._chunk_seed(plan.request, block_index),
                params=params,
            )
            closes = ohlcv["close"]
            if closes.size != block_n:
                raise SyntheticValidationError("model returned invalid block size")
            timestamps = generate_timestamps(
                timestamp_cursor,
                plan.context.interval,
                block_n,
                trading_hours=plan.context.trading_hours,
            )
            yield _OhlcvBlock(
                block_index=block_index,
                block_start=block_start,
                timestamps=timestamps,
                ohlcv=ohlcv,
            )
            last_close = float(closes[-1])
            timestamp_cursor = timestamps[-1] + plan.interval_step

    def _generate_frame_from_plan(self, plan: _RunPlan, *, compact_identity: bool = False):
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency boundary
            raise SyntheticValidationError("pandas is required for generate_frame") from exc

        frames = [
            self._frame_from_block(
                plan,
                block,
                offset_start=0,
                chunk_n=len(block.timestamps),
                compact_identity=compact_identity,
            )
            for block in self._iter_ohlcv_blocks(plan)
        ]
        if not frames:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "symbol", "venue", "dataset", "interval"])
        if len(frames) == 1:
            return frames[0]
        return pd.concat(frames, ignore_index=True)

    def _frame_from_block(
        self,
        plan: _RunPlan,
        block: _OhlcvBlock,
        *,
        offset_start: int,
        chunk_n: int,
        compact_identity: bool,
    ):
        try:
            import pandas as pd
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency boundary
            raise SyntheticValidationError("pandas is required for generate_frame") from exc

        offset_end = offset_start + chunk_n
        frame = pd.DataFrame(
            {
                "timestamp": pd.to_datetime(block.timestamps[offset_start:offset_end], utc=True),
                "open": block.ohlcv["open"][offset_start:offset_end],
                "high": block.ohlcv["high"][offset_start:offset_end],
                "low": block.ohlcv["low"][offset_start:offset_end],
                "close": block.ohlcv["close"][offset_start:offset_end],
                "volume": block.ohlcv["volume"][offset_start:offset_end],
            }
        )
        identity = self._identity_for_plan(plan)
        for column, value in identity.items():
            frame[column] = value
        if compact_identity:
            for column in identity:
                frame[column] = frame[column].astype("category")
        return frame

    def _identity_for_plan(self, plan: _RunPlan) -> dict[str, str | None]:
        return {
            "symbol": plan.context.symbol,
            "venue": plan.context.venue,
            "dataset": plan.context.dataset,
            "interval": plan.request.interval,
        }

    def _normalize_request(self, request: SyntheticRequest) -> SyntheticRequest:
        if request.start_ts.tzinfo is None:
            request = replace(request, start_ts=request.start_ts.replace(tzinfo=timezone.utc))
        else:
            request = replace(request, start_ts=request.start_ts.astimezone(timezone.utc))

        if request.start_price <= 0:
            raise SyntheticValidationError("start_price must be positive")
        return request

    def _resolve_run_context(self, request: SyntheticRequest) -> tuple[SyntheticContext, int]:
        context = self._resolve_context(request)
        n_bars = self._resolve_n_bars(request, context)
        return context, n_bars

    def _resolve_n_bars(self, request: SyntheticRequest, context: SyntheticContext) -> int:
        n_bars = bars_from_span(
            request.start_ts,
            request.interval,
            n_bars=request.n_bars,
            end_ts=request.end_ts,
            duration=request.duration,
            trading_hours=context.trading_hours,
        )
        if n_bars <= 0:
            raise SyntheticValidationError("n_bars must be greater than 0")
        return n_bars

    def _resolve_context(self, request: SyntheticRequest) -> SyntheticContext:
        if request.symbol is None:
            return SyntheticContext(
                symbol=None,
                venue=request.venue,
                dataset=request.dataset or request.model_params.get("dataset"),
                interval=request.interval,
                asset_class=request.model_params.get("asset_class", "generic"),
                trading_hours=request.model_params.get("trading_hours", "24x7"),
                timezone=request.model_params.get("timezone", "UTC"),
            )

        key = (request.symbol, request.venue, request.dataset, request.interval)
        if key in self._context_cache:
            return self._context_cache[key]

        if self.context_provider is None:
            raise SyntheticValidationError("context_provider is required to resolve symbol-aware context")

        if hasattr(self.context_provider, "resolve"):
            context = self.context_provider.resolve(
                symbol=request.symbol,
                venue=request.venue,
                dataset=request.dataset,
                interval=request.interval,
            )
        elif callable(self.context_provider):
            context = self.context_provider(
                symbol=request.symbol,
                venue=request.venue,
                dataset=request.dataset,
                interval=request.interval,
            )
        else:
            raise SyntheticValidationError("context_provider must be callable or expose resolve")
        self._context_cache[key] = context
        return context

    def _validate_context(self, context: SyntheticContext, request: SyntheticRequest) -> None:
        requested_hours = request.model_params.get("trading_hours")
        if requested_hours is not None and requested_hours != context.trading_hours:
            raise SyntheticContextError(
                f"request trading_hours='{requested_hours}' conflicts with "
                f"context trading_hours='{context.trading_hours}'"
            )

        supported = getattr(self.model, "required_trading_hours", None)
        if not supported:
            return

        candidate = context.trading_hours
        if candidate not in supported:
            raise SyntheticContextError(
                f"model '{self.model.model_name}' does not support trading_hours='{candidate}'. "
                f"Supported: {supported}"
            )

    def _run_seed(self, request: SyntheticRequest) -> int:
        return self._chunk_seed(request, 0)

    def _chunk_seed(self, request: SyntheticRequest, chunk_index: int) -> int:
        key = (
            "qik.synthetic.seed",
            self.seed,
            request.model,
            request.seed,
            request.start_ts,
            request.interval,
            request.start_price,
            request.n_bars,
            request.duration,
            request.end_ts,
            _normalized_model_params(request),
            chunk_index,
        )
        return _seed_to_int(repr(key).encode("utf-8"))

    def _request_id(self, request: SyntheticRequest) -> str:
        return _stable_id(
            "request",
            request.model,
            request.symbol,
            request.start_ts.isoformat(),
            request.interval,
            request.start_price,
            request.n_bars,
            request.duration,
            request.end_ts,
            request.seed,
            _normalized_model_params(request),
        )

    def _run_id(self, request: SyntheticRequest) -> str:
        return _stable_id(
            "run",
            self.seed,
            request.model,
            request.symbol,
            request.start_ts.isoformat(),
            request.interval,
            request.start_price,
            request.n_bars,
            request.duration,
            request.end_ts,
            request.seed,
            _normalized_model_params(request),
        )

    def _build_metadata(self, request: SyntheticRequest, context: SyntheticContext) -> SyntheticMetadata:
        context_id = context.symbol or context.dataset or context.venue or request.symbol or "synthetic"
        return SyntheticMetadata(
            request_id=self._request_id(request),
            run_id=self._run_id(request),
            seed=self._run_seed(request),
            model_name=self.model.model_name,
            context_id=context_id,
        )


def _normalized_model_params(request: SyntheticRequest) -> tuple[tuple[str, object], ...]:
    return tuple(sorted((str(key), _stable_param_repr(value)) for key, value in request.model_params.items()))


def _stable_param_repr(value: object) -> object:
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            return ("ndarray", str(value.dtype), tuple(value.shape), tuple(_stable_param_repr(item) for item in value.ravel(order="C")))
        array = np.ascontiguousarray(value)
        digest = blake2b(array.view(np.uint8), digest_size=16).hexdigest()
        return ("ndarray", str(array.dtype), tuple(array.shape), digest)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _stable_param_repr(item)) for key, item in value.items()))
    if isinstance(value, tuple):
        return tuple(_stable_param_repr(item) for item in value)
    if isinstance(value, list):
        return ("list", tuple(_stable_param_repr(item) for item in value))
    if isinstance(value, set):
        return ("set", tuple(sorted((_stable_param_repr(item) for item in value), key=repr)))
    return repr(value)
