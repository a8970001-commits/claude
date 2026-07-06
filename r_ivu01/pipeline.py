"""Orchestrates Steps 1-6 (Section 4) and Gates G1-G5 (Section 5).

This module wires the pure building blocks in indicators/, events/, and
stats/ together. It does NOT itself decide statistical thresholds — those
all live in config.py and stats/gates.py. Where a step needs real market
data (finlab TW panel, yfinance US panel, corporate-action calendars), the
call sites are marked TODO; supply the concrete DataFrames built from
data/taiwan.py / data/us.py and this module runs unmodified.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from r_ivu01.config import (
    ATR_PERIOD,
    FORWARD_WINDOWS,
    GLOBAL_WARMUP_BARS,
    N_DECILES,
    PRIMARY_H,
    IS_END,
    tw_roundtrip_cost,
    us_roundtrip_cost,
)
from r_ivu01.events.breakout import detect_breakouts, dedup_events
from r_ivu01.events.exclusions import global_warmup_exclusion
from r_ivu01.indicators.ace import run_ace_white_line, simulate_exit
from r_ivu01.indicators.forward import forward_indicators
from r_ivu01.indicators.rvol import rvol as compute_rvol
from r_ivu01.stats.binning import assign_deciles, fit_decile_edges
from r_ivu01.stats.bootstrap import assign_iso_week
from r_ivu01.stats.curve_fit import select_kernel
from r_ivu01.stats.gates import (
    GateResult,
    g1_shape_gate,
    g2_time_stability,
    g3_oos_validity,
    g4_economic_significance,
    g5_cross_market_consistency,
)


@dataclass
class TickerPriceData:
    ticker: str
    close: pd.Series
    high: pd.Series
    low: pd.Series
    volume: pd.Series


@dataclass
class MarketEvents:
    market: str  # "TW" | "US"
    events: pd.DataFrame  # one row per surviving event, all tickers pooled


def build_events_for_ticker(
    price: TickerPriceData,
    extra_exclusion_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """3.1 breakout detection -> 2.3/global warmup exclusion -> dedup ->
    3.2 RVOL -> 3.3 layer-1 forward indicators, for ONE ticker.

    `extra_exclusion_mask`: caller-supplied OR of all market-specific
    exclusions (limit-lock, ex-div, share-count-change, disposition,
    liquidity/price screen for TW; split window for US) — these depend on
    data this scaffold doesn't fetch (see events/exclusions.py stubs).
    """

    raw_candidates = detect_breakouts(price.close, price.high)

    excluded = global_warmup_exclusion(raw_candidates)
    if extra_exclusion_mask is not None:
        excluded = excluded | extra_exclusion_mask.reindex(raw_candidates.index, fill_value=False)

    candidates = raw_candidates & ~excluded
    kept = dedup_events(candidates)

    if not kept.any():
        return pd.DataFrame()

    rvol = compute_rvol(price.volume)
    fwd = forward_indicators(price.close, price.high, price.low, ATR_PERIOD, FORWARD_WINDOWS)

    event_positions = np.flatnonzero(kept.to_numpy())
    out = pd.DataFrame(
        {
            "ticker": price.ticker,
            "date": price.close.index[event_positions],
            "position": event_positions,
            "rvol": rvol.iloc[event_positions].to_numpy(),
        }
    )
    for col in fwd.columns:
        out[col] = fwd[col].iloc[event_positions].to_numpy()

    return out.dropna(subset=[f"R_fwd_{PRIMARY_H}"])


def pool_events(per_ticker_events: list[pd.DataFrame], market: str) -> MarketEvents:
    events = pd.concat(per_ticker_events, ignore_index=True) if per_ticker_events else pd.DataFrame()
    if not events.empty:
        events["iso_week"] = assign_iso_week(events["date"])
        events["sample"] = np.where(events["date"] <= pd.Timestamp(IS_END), "in_sample", "oos")
    return MarketEvents(market=market, events=events)


def run_layer1_shape_pipeline(market_events: MarketEvents, n_resamples: int = 1000) -> dict:
    """Steps 1-4 + G1/G2 for the PRIMARY combination (market x h=5 x
    trimmed-mean R_fwd). Secondary combinations (other h, other metrics) are
    computed the same way but must never feed a gate decision (E-06)."""

    events = market_events.events
    in_sample = events[events["sample"] == "in_sample"]

    edges = fit_decile_edges(in_sample["rvol"])
    binning = assign_deciles(events["rvol"], edges)
    events = events.assign(decile=binning.labels)

    value_col = f"R_fwd_{PRIMARY_H}"

    g1 = g1_shape_gate(
        events.dropna(subset=["decile"]),
        cluster_col="iso_week",
        decile_col="decile",
        value_col=value_col,
        n_bins=N_DECILES,
        n_resamples=n_resamples,
    )

    in_sample_binned = events[(events["sample"] == "in_sample")].dropna(subset=["decile"])
    midpoint = in_sample["date"].median()
    first_half = in_sample_binned[in_sample_binned["date"] <= midpoint]
    second_half = in_sample_binned[in_sample_binned["date"] > midpoint]

    def peak_mu(sub: pd.DataFrame) -> float:
        bin_means = sub.groupby("decile")[value_col].mean().reindex(range(1, N_DECILES + 1))
        return float(bin_means.idxmax()) if not bin_means.isna().all() else np.nan

    g2 = g2_time_stability(peak_mu(first_half), peak_mu(second_half))

    kernel_fit = None
    if g1.passed:
        bin_stats = events.dropna(subset=["decile"]).groupby("decile")[value_col].agg(["mean", "sem", "count"])
        weights = 1.0 / np.maximum(bin_stats["sem"].to_numpy() ** 2, 1e-8)
        kernel_fit = select_kernel(
            rvol=bin_stats.index.to_numpy(dtype=float),
            target=bin_stats["mean"].to_numpy(),
            weights=weights,
        )

    return {"edges": edges, "events": events, "g1": g1, "g2": g2, "kernel_fit": kernel_fit}


def run_layer2_ace_simulation(
    events: pd.DataFrame,
    price_by_ticker: dict[str, TickerPriceData],
) -> pd.DataFrame:
    """Step 5: run the ACE white-line engine per ticker (full history, Pine
    `var`-style persistence) and simulate the exit for every event."""

    rows = []
    for ticker, group in events.groupby("ticker"):
        price = price_by_ticker[ticker]
        ace_result = run_ace_white_line(price.high, price.low, price.close)

        for _, event in group.iterrows():
            entry_pos = int(event["position"])
            entry_price = float(price.close.iloc[entry_pos])
            exit_result = simulate_exit(ace_result, price.close, entry_pos)

            row = dict(event)
            row["ace_dir_at_entry"] = ace_result.ace_dir.iloc[entry_pos]
            row["exit_reason"] = exit_result.exit_reason
            if exit_result.exit_price is not None:
                atr_denom = event["atr_t_minus_1"]
                row["realized_R"] = (exit_result.exit_price - entry_price) / atr_denom
                row["pct_return"] = (exit_result.exit_price - entry_price) / entry_price
            else:
                row["realized_R"] = np.nan
                row["pct_return"] = np.nan
            rows.append(row)

    return pd.DataFrame(rows)


def run_g4_gate(layer2_events: pd.DataFrame, kernel_fit, edges, market: str) -> GateResult:
    """G4 judged on layer 2, high-weight group defined by the Step-4 kernel
    w(RVOL) >= 0.7, excluding events excluded at entry (ace_dir != 1)."""

    valid = layer2_events.dropna(subset=["pct_return"])
    if kernel_fit is None or valid.empty:
        return GateResult("G4_economic_significance", False, {"reason": "no kernel fit or no valid events"})

    from r_ivu01.stats.curve_fit import gamma_kernel, log_gaussian_kernel

    fn = log_gaussian_kernel if kernel_fit.name == "log_gaussian" else gamma_kernel
    w = fn(valid["rvol"].to_numpy(), *kernel_fit.params)
    high_weight_group = valid[w >= 0.7]

    cost = tw_roundtrip_cost() if market == "TW" else us_roundtrip_cost()
    return g4_economic_significance(high_weight_group["pct_return"].to_numpy(), cost)


def run_g3_gate(
    layer1_events: pd.DataFrame,
    kernel_fit,
    n_resamples: int = 1000,
    random_state: int | None = None,
) -> GateResult:
    """Step 6 / G3: in-sample-fitted kernel weights applied out-of-sample
    (2023-2026). Spread = high-weight (w>=0.7) minus low-weight (w<0.3) group
    mean h=5 forward return; CI via the same cluster-bootstrap method (E-05),
    cluster = OOS event ISO calendar week."""

    from r_ivu01.stats.curve_fit import gamma_kernel, log_gaussian_kernel
    from r_ivu01.stats.bootstrap import cluster_bootstrap_resample

    oos = layer1_events[layer1_events["sample"] == "oos"].dropna(subset=[f"R_fwd_{PRIMARY_H}"])
    if kernel_fit is None or oos.empty:
        return GateResult("G3_oos_validity", False, {"reason": "no kernel fit or no OOS events"})

    fn = log_gaussian_kernel if kernel_fit.name == "log_gaussian" else gamma_kernel
    oos = oos.assign(w_vol=fn(oos["rvol"].to_numpy(), *kernel_fit.params))

    def spread_statistic(sample: pd.DataFrame) -> float:
        high = sample[sample["w_vol"] >= 0.7][f"R_fwd_{PRIMARY_H}"]
        low = sample[sample["w_vol"] < 0.3][f"R_fwd_{PRIMARY_H}"]
        if high.empty or low.empty:
            return np.nan
        return float(high.mean() - low.mean())

    samples = cluster_bootstrap_resample(oos, "iso_week", n_resamples, spread_statistic, random_state)
    samples = np.array([s for s in samples if not np.isnan(s)])
    if samples.size == 0:
        return GateResult("G3_oos_validity", False, {"reason": "all resamples empty a group"})

    return g3_oos_validity(samples)
