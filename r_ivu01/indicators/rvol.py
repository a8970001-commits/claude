"""RVOL — 3.2. Denominator is t-1 cutoff to avoid self-contamination."""

from __future__ import annotations

import pandas as pd

from r_ivu01.config import RVOL_WINDOW


def rvol(volume: pd.Series, window: int = RVOL_WINDOW) -> pd.Series:
    avg_volume_prior = volume.rolling(window).mean().shift(1)
    return volume / avg_volume_prior
