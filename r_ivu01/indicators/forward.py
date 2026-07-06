"""Layer 1 — shape identification forward indicators (3.3, F-01).

R_fwd[h] = (close[t+h] - close[t]) / ATR(14)[t-1]   (denominator fixed at t-1,
            NOT ATR(14)[t] — the event day's own TR would enter the RMA with
            weight 1/14 and correlate with RVOL, bending a monotone truth into
            a false inverted-U).
win[h]   = 1 if close[t+h] > close[t] else 0        (binary, no denominator,
            unaffected by F-01).
"""

from __future__ import annotations

import pandas as pd

from r_ivu01.config import FORWARD_WINDOWS
from r_ivu01.indicators.atr import atr as compute_atr


def forward_indicators(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr_period: int,
    windows: tuple[int, ...] = FORWARD_WINDOWS,
) -> pd.DataFrame:
    atr_series = compute_atr(high, low, close, atr_period)
    atr_t_minus_1 = atr_series.shift(1)

    out = {}
    for h in windows:
        fwd_close = close.shift(-h)
        out[f"R_fwd_{h}"] = (fwd_close - close) / atr_t_minus_1
        out[f"win_{h}"] = (fwd_close > close).astype("float")
    out["atr_t_minus_1"] = atr_t_minus_1
    return pd.DataFrame(out, index=close.index)
