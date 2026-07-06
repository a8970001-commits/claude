"""Breakout event definition — 3.1 (裁決C: generic version, BEST-aligned version
deferred to a stage-2 spec). LOCKED — do not modify without a new spec version."""

from __future__ import annotations

import numpy as np
import pandas as pd

from r_ivu01.config import BREAKOUT_LOOKBACK_HIGH, DEDUP_MIN_GAP_DAYS, SMA_LONG, SMA_MID


def detect_breakouts(close: pd.Series, high: pd.Series) -> pd.Series:
    """Returns a boolean Series, True on raw breakout-candidate days (before
    2.3 exclusion rules and before dedup are applied)."""

    prior_high_20 = high.shift(1).rolling(BREAKOUT_LOOKBACK_HIGH).max()
    sma50 = close.rolling(SMA_MID).mean()
    sma200 = close.rolling(SMA_LONG).mean()

    return (close > prior_high_20) & (close > sma50) & (sma50 > sma200)


def dedup_events(is_event: pd.Series, min_gap_days: int = DEDUP_MIN_GAP_DAYS) -> pd.Series:
    """Same-ticker events within min_gap_days of a kept event are dropped;
    only the first event of a cluster survives."""

    kept = pd.Series(False, index=is_event.index)
    last_kept_pos = -min_gap_days - 1
    positions = np.flatnonzero(is_event.to_numpy())
    for pos in positions:
        if pos - last_kept_pos >= min_gap_days:
            kept.iloc[pos] = True
            last_kept_pos = pos
    return kept
