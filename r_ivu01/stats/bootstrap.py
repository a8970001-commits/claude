"""Cluster bootstrap by ISO calendar week — v1.3/E-05, shared by G1 and G3.

Rationale (spec Step 3): breakout events cluster heavily on market-wide
attack days and share beta in their forward returns; iid per-event resampling
would understate CI width. Resampling unit is the ISO week, not the event.
"""

from __future__ import annotations

from typing import Callable, TypeVar

import numpy as np
import pandas as pd

T = TypeVar("T")


def assign_iso_week(dates: pd.Series) -> pd.Series:
    iso = pd.DatetimeIndex(dates).isocalendar()
    return (iso["year"].astype(str) + "-W" + iso["week"].astype(str).str.zfill(2)).reset_index(drop=True)


def cluster_bootstrap_resample(
    df: pd.DataFrame,
    cluster_col: str,
    n_resamples: int,
    statistic_fn: Callable[[pd.DataFrame], T],
    random_state: int | None = None,
) -> list[T]:
    """Resample whole clusters (ISO weeks) with replacement, n_resamples times,
    applying `statistic_fn` to each resampled frame. Frozen bin edges must be
    passed into statistic_fn by the caller (do not refit edges per resample)."""

    rng = np.random.default_rng(random_state)
    clusters = df[cluster_col].unique()
    grouped = {c: idx for c, idx in df.groupby(cluster_col).groups.items()}

    results = []
    for _ in range(n_resamples):
        sampled_clusters = rng.choice(clusters, size=len(clusters), replace=True)
        idx = np.concatenate([grouped[c].to_numpy() for c in sampled_clusters])
        resampled = df.loc[idx]
        results.append(statistic_fn(resampled))
    return results
