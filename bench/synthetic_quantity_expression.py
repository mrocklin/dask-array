"""Standalone dask_array workload shaped like the research quantity DAG.

Copied verbatim from the statarb research repo (pipeline/), where it was built
as a standalone, dependency-light stand-in for the research quantity DAG.  It
uses only numpy/dask/dask_array, so it runs from anywhere; graph shape:

* many delayed day/source leaves
* concatenation over time
* shared derived ancestors
* pointwise algebra, masks, logs, clips, shifts
* rolling/overlap windows
* cumulative scans
* cross-sectional reductions over assets
* optional rechunk boundaries
* many stacked output quantities

Examples:

    python bench/synthetic_quantity_expression.py
    python bench/synthetic_quantity_expression.py 4
    python bench/synthetic_quantity_expression.py 2

The reusable public function is:

    from synthetic_quantity_expression import synthetic_quantity_array

    x = synthetic_quantity_array(complexity=2)  # shape=(quantity, time, asset)
"""

from __future__ import annotations

import argparse
import time
import zlib
from dataclasses import dataclass

from dask import delayed
import numpy as np

import dask_array as da


BASE_NAMES = [
    "bid_price",
    "ask_price",
    "open",
    "close",
    "high",
    "low",
    "vol",
    "vol_buy",
    "vol_sell",
    "vol_dol",
    "vol_dol_buy",
    "vol_dol_sell",
    "count",
    "open_interest",
    "mark_price",
    "index_price",
    "funding_rate",
    "long_short_ratio",
    "liquidations_buy",
    "liquidations_sell",
    "amount_ssq",
    "amount_sum",
    "quote_pressure",
    "trade_pressure",
    "spread_source",
    "depth_bid",
    "depth_ask",
    "imbalance",
    "volatility_seed",
    "carry_seed",
    "basis_seed",
    "external_index",
]


@dataclass(frozen=True)
class SyntheticQuantityWorkload:
    stack: da.Array
    targets: list[str]
    arrays: dict[str, da.Array]


def _load_day(name: str, day: int, n_time: int, n_assets: int, seed: int) -> np.ndarray:
    token = zlib.crc32(f"{seed}:{day}:{name}".encode("utf-8"))
    rng = np.random.default_rng(token)
    t = np.arange(n_time, dtype=np.float64)[:, None]
    a = np.arange(n_assets, dtype=np.float64)[None, :]
    noise = rng.standard_normal((n_time, n_assets))

    if "price" in name or name in {"open", "close", "high", "low"}:
        data = (
            100.0 + 0.02 * day + 0.0002 * t + 0.05 * a + 0.2 * np.sin((t + 17 * a + token % 31) / 700.0) + 0.03 * noise
        )
        if name == "ask_price":
            data = data + 0.05 + 0.002 * (a % 5)
        elif name == "bid_price":
            data = data - 0.05 - 0.002 * (a % 5)
        elif name == "high":
            data = data + np.abs(noise) * 0.08
        elif name == "low":
            data = data - np.abs(noise) * 0.08
    elif "vol" in name or "amount" in name or name == "count":
        data = np.exp(10.0 + 0.2 * np.sin((t + a) / 300.0) + 0.15 * noise)
        if "buy" in name:
            data *= 0.52
        elif "sell" in name:
            data *= 0.48
        if "ssq" in name:
            data = data * data * 1e-6
    else:
        data = 0.2 * np.sin((t + token % 97) / 250.0) + 0.1 * np.cos((a + token % 13) / 5.0) + 0.05 * noise

    missing = rng.random((n_time, n_assets)) < 0.01
    data = data.astype("float64", copy=False)
    data[missing] = np.nan
    return data


def _rolling_sum_block(block: np.ndarray, window: int) -> np.ndarray:
    data = np.nan_to_num(block, nan=0.0)
    csum = np.cumsum(data, axis=0)
    if window >= len(block):
        previous = np.zeros_like(csum)
    else:
        head = np.zeros((window, block.shape[1]), dtype=block.dtype)
        previous = np.concatenate([head, csum[:-window]], axis=0)
    return csum - previous


def _source(
    name: str,
    days: int,
    n_time: int,
    n_assets: int,
    asset_chunk: int | None,
    seed: int,
) -> da.Array:
    pieces = []
    for day in range(days):
        piece = da.from_delayed(
            delayed(_load_day)(name, day, n_time, n_assets, seed),
            shape=(n_time, n_assets),
            dtype="float64",
        )
        if asset_chunk is not None and asset_chunk < n_assets:
            piece = piece.rechunk((n_time, asset_chunk))
        pieces.append(piece)
    return da.concatenate(pieces, axis=0)


def fillna(x: da.Array, value: float = 0.0) -> da.Array:
    return da.where(da.isnan(x), value, x)


def safe_divide(x: da.Array, y: da.Array) -> da.Array:
    return x / da.where(da.abs(y) > 1e-12, y, np.nan)


def shift_time(x: da.Array, periods: int) -> da.Array:
    if periods == 0:
        return x
    if abs(periods) >= x.shape[0]:
        return da.full_like(x, np.nan)

    pad = da.full(
        (abs(periods), x.shape[1]),
        np.nan,
        chunks=((abs(periods),), x.chunks[1]),
    )
    if periods > 0:
        return da.concatenate([pad, x[:-periods]], axis=0)
    return da.concatenate([x[-periods:], pad], axis=0)


def rolling_sum(x: da.Array, window: int) -> da.Array:
    window = max(1, int(window))
    return da.map_overlap(
        _rolling_sum_block,
        x,
        depth=(window - 1, 0),
        boundary=np.nan,
        dtype=x.dtype,
        trim=True,
        window=window,
    )


def rolling_mean(x: da.Array, window: int) -> da.Array:
    valid = da.where(da.isnan(x), np.nan, 1.0)
    return safe_divide(rolling_sum(x, window), rolling_sum(valid, window))


def cumulative_smooth(x: da.Array, window: int) -> da.Array:
    raw = da.cumsum(fillna(x, 0.0), axis=0)
    lagged = shift_time(raw, max(1, int(window)))
    return safe_divide(raw - fillna(lagged, 0.0), float(max(1, int(window))))


def weighted_mean(x: da.Array, weight: da.Array) -> da.Array:
    valid_weight = da.where(da.isnan(x) | da.isnan(weight), 0.0, da.abs(weight))
    numerator = da.sum(fillna(x, 0.0) * valid_weight, axis=1, keepdims=True)
    denominator = da.sum(valid_weight, axis=1, keepdims=True)
    return numerator / da.where(denominator > 0, denominator, np.nan)


def mean_adjust(x: da.Array, weight: da.Array) -> da.Array:
    return x - weighted_mean(x, weight)


def zscore(x: da.Array, weight: da.Array) -> da.Array:
    centered = mean_adjust(x, weight)
    variance = weighted_mean(centered * centered, weight)
    return centered / da.sqrt(variance + 1e-12)


def beta_adjust(x: da.Array, other: da.Array, weight: da.Array) -> da.Array:
    x_centered = mean_adjust(x, weight)
    other_centered = mean_adjust(other, weight)
    covariance = weighted_mean(x_centered * other_centered, weight)
    variance = weighted_mean(other_centered * other_centered, weight)
    beta = covariance / da.where(variance > 1e-12, variance, np.nan)
    return x_centered - beta * other_centered


def martingale_sum(x: da.Array, horizon: int, step: int) -> da.Array:
    out = x
    total_steps = max(1, horizon // max(1, step))
    for i in range(1, min(total_steps, 8)):
        candidate = shift_time(x, -i * step)
        valid = (out > -0.08) & (out < 0.08)
        out = out + da.where(valid, candidate, 0.0) * (0.92**i)
    return out


def _build_expression(
    *,
    days: int = 7,
    n_time_per_day: int = 8640,
    assets: int = 64,
    asset_chunk: int | None = 16,
    n_base: int = 32,
    families: int = 96,
    targets: int = 0,
    seed: int = 0,
    rechunk_every: int = 5,
) -> SyntheticQuantityWorkload:
    """Build a lazy synthetic quantity expression.

    ``targets=0`` means keep every generated target.  ``families`` controls graph
    breadth; each family contributes three final quantities plus shared
    intermediates.
    """
    if days < 1:
        raise ValueError("days must be positive")
    if n_time_per_day < 4:
        raise ValueError("n_time_per_day must be at least 4")
    if assets < 2:
        raise ValueError("assets must be at least 2")
    if asset_chunk is not None and asset_chunk < 1:
        raise ValueError("asset_chunk must be positive or None")
    if n_base < 8:
        raise ValueError("n_base must be at least 8")
    if families < 1:
        raise ValueError("families must be positive")
    if targets < 0:
        raise ValueError("targets must be non-negative")

    required = [
        "bid_price",
        "ask_price",
        "vol_dol",
        "funding_rate",
        "open",
        "close",
        "high",
        "low",
    ]
    base_names = list(dict.fromkeys(required + BASE_NAMES))
    while len(base_names) < n_base:
        base_names.append(f"extra_source_{len(base_names):03d}")
    base_names = base_names[: max(n_base, len(required))]

    arrays = {name: _source(name, days, n_time_per_day, assets, asset_chunk, seed) for name in base_names}
    first = arrays[base_names[0]]
    one = da.ones_like(first)

    bid = fillna(arrays["bid_price"], 100.0)
    ask = fillna(arrays["ask_price"], 100.1)
    mid = (bid + ask) * 0.5
    spread = da.clip(safe_divide(ask - bid, mid), 1e-7, 0.1)
    volume = fillna(arrays["vol_dol"], 0.0)
    adv = rolling_sum(volume, min(days * n_time_per_day // 2, 43200)) * 0.2 + 1.0
    adv_1d = rolling_sum(volume, min(n_time_per_day, 8640)) + 1.0
    log_adv = da.log(adv_1d + 1.0) / np.log(10.0)
    spread_weight = da.clip((0.002 - spread) / 0.002, 0.0, 1.0)
    weight = da.clip((log_adv - 6.5) / 2.0, 0.05, 1.0) * spread_weight
    market_horizon = max(1, min(60, days * n_time_per_day // 4))
    market = mean_adjust(da.log(shift_time(mid, market_horizon) / mid), weight)
    time_axis = da.arange(days * n_time_per_day, chunks=first.chunks[0])[:, None]
    periodic = da.sin(time_axis / 360.0) + da.cos(time_axis / 1800.0)
    carry = fillna(arrays["funding_rate"], 0.0) * 1e-4
    adjustment = da.exp(-da.cumsum(carry, axis=0))

    arrays.update(
        {
            "mid": mid,
            "spread": spread,
            "adv": adv,
            "adv_1d": adv_1d,
            "weight": weight,
            "market": market,
            "periodic": periodic + 0.0 * one,
            "adjustment": adjustment,
        }
    )

    windows = [3, 18, 60, 180, 540, 2160, 7200]
    horizons = [1, 3, 12, 60, 300, 1800]
    max_window = max(2, min(days * n_time_per_day // 2, n_time_per_day))
    windows = [max(2, min(w, max_window)) for w in windows]
    horizons = [max(1, min(h, days * n_time_per_day // 4)) for h in horizons]

    output_names: list[str] = []
    outputs: list[da.Array] = []
    source_cycle = [name for name in base_names if name not in {"bid_price", "ask_price"}]

    families_to_build = families if not targets else min(families, (targets + 2) // 3)
    for i in range(families_to_build):
        window = windows[i % len(windows)]
        horizon = horizons[i % len(horizons)]
        source_name = source_cycle[i % len(source_cycle)]
        secondary_name = source_cycle[(i * 7 + 3) % len(source_cycle)]

        source = fillna(arrays[source_name], 0.0)
        secondary = fillna(arrays[secondary_name], 0.0)
        future_mid = shift_time(mid * adjustment, -horizon)
        past_mid = shift_time(mid, horizon)
        ret = da.log(da.clip(safe_divide(future_mid, mid), 1e-4, 1e4))
        hist_ret = -da.log(da.clip(safe_divide(past_mid, mid), 1e-4, 1e4))
        flow = da.log1p(da.abs(source)) - da.log1p(rolling_mean(da.abs(secondary), window))
        flow = da.clip(flow, -6.0, 6.0)

        trend = rolling_mean(ret + 0.2 * hist_ret + 0.01 * flow, window)
        shock = cumulative_smooth(mean_adjust(trend, weight), window)
        signal = zscore(shock + 0.03 * arrays["periodic"], weight)

        if i % 2 == 0:
            signal = beta_adjust(signal, market, weight)
        if i % 3 == 0:
            signal = martingale_sum(signal, horizon=max(2 * horizon, window), step=max(1, horizon))
        if rechunk_every and i % rechunk_every == 0:
            asset_block = assets if asset_chunk is None else min(asset_chunk, assets)
            signal = signal.rechunk((min(days * n_time_per_day, 2 * n_time_per_day), asset_block))

        masked = da.where((spread < 0.03) & (adv > 1.0), signal, np.nan)
        risk = rolling_mean(signal * signal, max(2, window // 3))
        normalized = da.clip(masked / da.sqrt(risk + 1e-6), -8.0, 8.0)

        for suffix, value in (
            ("signal", signal),
            ("masked", masked),
            ("normalized", normalized),
        ):
            name = f"q{i:03d}_{suffix}"
            arrays[name] = value
            output_names.append(name)
            outputs.append(value)

    if targets:
        output_names = output_names[:targets]
        outputs = outputs[:targets]
    stack = da.stack(outputs, axis=0)
    return SyntheticQuantityWorkload(stack=stack, targets=output_names, arrays=arrays)


def synthetic_quantity_workload(complexity: int = 1) -> SyntheticQuantityWorkload:
    """Return a synthetic quantity workload with one scale knob.

    ``complexity=1`` is a quick local smoke test.  Larger values increase time
    rows, assets, source arrays, and quantity families together while preserving
    the same expression shape.
    """
    if complexity < 1:
        raise ValueError("complexity must be positive")

    days = min(7, complexity + 1)
    n_time_per_day = 720 * complexity
    assets = 16 * complexity
    asset_chunk = max(4, min(16, assets // 4))
    n_base = min(len(BASE_NAMES), 8 + 3 * complexity)
    families = 24 * complexity
    return _build_expression(
        days=days,
        n_time_per_day=n_time_per_day,
        assets=assets,
        asset_chunk=asset_chunk,
        n_base=n_base,
        families=families,
    )


def synthetic_quantity_array(complexity: int = 1) -> da.Array:
    """Return the stacked synthetic quantity array.

    The result has shape ``(quantity, time, asset)`` and is fully lazy.
    """
    return synthetic_quantity_workload(complexity).stack


def build_reductions(stack: da.Array) -> tuple[da.Array, da.Array]:
    finite = da.where(da.isnan(stack), 0.0, 1.0)
    counts = da.sum(finite, axis=(1, 2))
    means = da.sum(fillna(stack, 0.0), axis=(1, 2)) / da.where(counts > 0, counts, np.nan)
    return means, counts


def _print_summary(workload: SyntheticQuantityWorkload) -> None:
    stack = workload.stack
    print(f"targets: {len(workload.targets)}")
    print(f"shape:   {stack.shape}")
    print(f"chunks:  {stack.chunks}")
    print(f"blocks:  {stack.numblocks}")
    print(f"first targets: {', '.join(workload.targets[:8])}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("complexity", nargs="?", type=int, default=1)
    args = parser.parse_args(argv)

    t0 = time.perf_counter()
    workload = synthetic_quantity_workload(args.complexity)
    print(f"synthetic_quantity_workload: {time.perf_counter() - t0:.3f}s")
    _print_summary(workload)


if __name__ == "__main__":
    main()
