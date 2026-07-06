"""2.3 exclusion rules (TW) + F-08 amendments + US split window.

These need corporate-action / disposition-flag data that isn't available in
this scaffold (finlab disposition markers, ex-dividend calendars, share-count
change events). Each function documents its exact required input shape so the
data-wiring step can plug in real feeds without touching this logic.
"""

from __future__ import annotations

import pandas as pd

from r_ivu01.config import (
    DISPOSITION_VOL_SHRINK_PROXY,
    EX_DIVIDEND_EXCLUDE_DAYS_BEFORE,
    GLOBAL_WARMUP_BARS,
    LIMIT_UP_MAX_INTRADAY_RANGE,
    SHARE_CHANGE_EXCLUDE_DAYS_AFTER,
    SHARE_CHANGE_EXCLUDE_DAYS_AFTER_RESTORED,
    US_SPLIT_EXCLUDE_DAYS,
)


def limit_up_lock_exclusion(close: pd.Series, high: pd.Series, low: pd.Series, limit_price: pd.Series) -> pd.Series:
    """Rule 1: close == limit price AND intraday range < 1% -> excluded.

    `limit_price` must be precomputed per-ticker-per-day (TW daily limit-up
    price, e.g. prev_close * 1.10 subject to tick-size rounding) — not derived
    here, since the exact rounding convention is exchange-rule data, not a
    formula this factor owns.
    """
    intraday_range = (high - low) / close
    return (close == limit_price) & (intraday_range < LIMIT_UP_MAX_INTRADAY_RANGE)


def ex_dividend_exclusion(is_event: pd.Series, ex_dividend_dates: pd.DatetimeIndex) -> pd.Series:
    """Rule 2 (cash dividend only, no share-count change): exclude if any
    ex-dividend date falls within the 5 trading days before the event day."""

    excluded = pd.Series(False, index=is_event.index)
    if len(ex_dividend_dates) == 0:
        return excluded
    idx = is_event.index
    for ex_date in ex_dividend_dates:
        pos = idx.searchsorted(ex_date)
        window_start = max(0, pos - EX_DIVIDEND_EXCLUDE_DAYS_BEFORE)
        excluded.iloc[window_start:pos] = True
    return excluded


def share_count_change_exclusion(
    is_event: pd.Series,
    change_dates: pd.DatetimeIndex,
    volume_already_ex_adjusted: bool,
) -> pd.Series:
    """v1.3/E-08: rights issue / stock split / capital-reduction-restoration
    events. Exclusion window is AFTER the corporate action (RVOL's 20-day
    volume denominator mixes pre/post share-count volume for 20 trading days).

    If finlab's volume field is already ex-adjusted (pre-validation clause,
    2.3 rule 2), the window shrinks to +/-2 trading days, aligned with the US
    split rule.
    """
    excluded = pd.Series(False, index=is_event.index)
    if len(change_dates) == 0:
        return excluded
    idx = is_event.index
    after_days = (
        SHARE_CHANGE_EXCLUDE_DAYS_AFTER_RESTORED
        if volume_already_ex_adjusted
        else SHARE_CHANGE_EXCLUDE_DAYS_AFTER
    )
    for change_date in change_dates:
        pos = idx.searchsorted(change_date)
        if volume_already_ex_adjusted:
            window_start = max(0, pos - after_days)
        else:
            window_start = pos
        window_end = min(len(idx), pos + after_days + 1)
        excluded.iloc[window_start:window_end] = True
    return excluded


def disposition_exclusion(
    is_event: pd.Series,
    disposition_flag: pd.Series | None,
    volume_20d: pd.Series,
    volume_60d: pd.Series,
) -> pd.Series:
    """Rule 3: use finlab's disposition-stock flag when available; otherwise
    approximate with '20d avg volume shrunk >70% vs 60d avg volume' and mark
    the result as an approximation in the report."""
    if disposition_flag is not None:
        return disposition_flag.astype(bool)
    with pd.option_context("mode.use_inf_as_na", True):
        shrink = 1 - (volume_20d / volume_60d)
    return shrink > DISPOSITION_VOL_SHRINK_PROXY


def global_warmup_exclusion(is_event: pd.Series, listing_start_pos: int = 0) -> pd.Series:
    """Rule 4 (v1.3/E-03): first GLOBAL_WARMUP_BARS trading days per ticker
    are never event days. Shared by BOTH layers (H4 sample comformity)."""
    excluded = pd.Series(False, index=is_event.index)
    cutoff = listing_start_pos + GLOBAL_WARMUP_BARS
    excluded.iloc[:cutoff] = True
    return excluded


def us_split_exclusion(is_event: pd.Series, split_dates: pd.DatetimeIndex) -> pd.Series:
    """Rule 5 (US): +/- 2 trading days around a stock split date."""
    excluded = pd.Series(False, index=is_event.index)
    if len(split_dates) == 0:
        return excluded
    idx = is_event.index
    for split_date in split_dates:
        pos = idx.searchsorted(split_date)
        window_start = max(0, pos - US_SPLIT_EXCLUDE_DAYS)
        window_end = min(len(idx), pos + US_SPLIT_EXCLUDE_DAYS + 1)
        excluded.iloc[window_start:window_end] = True
    return excluded
