import numpy as np
import pandas as pd

from r_ivu01.indicators.ace import run_ace_white_line, simulate_exit


def _synthetic_trend(n=400, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2015-01-01", periods=n, freq="B")
    steps = rng.normal(0.05, 1.0, n)
    close = pd.Series(100 + np.cumsum(steps), index=idx)
    high = close + rng.uniform(0.1, 1.0, n)
    low = close - rng.uniform(0.1, 1.0, n)
    return close, high, low


def test_ace_trail_ratchets_up_only_while_in_uptrend():
    close, high, low = _synthetic_trend()
    result = run_ace_white_line(high, low, close)
    trail = result.trail_struct_l.dropna().to_numpy()
    dirs = result.ace_dir.dropna().to_numpy()

    # while continuously in aceDir==1, the trail must never decrease
    for i in range(1, len(trail)):
        if dirs[i] == 1 and dirs[i - 1] == 1:
            assert trail[i] >= trail[i - 1] - 1e-9


def test_ace_dir_flips_when_close_breaks_trail():
    close, high, low = _synthetic_trend()
    result = run_ace_white_line(high, low, close)
    close_v = close.to_numpy()
    trail_v = result.trail_struct_l.to_numpy()
    dirs = result.ace_dir.to_numpy()

    for t in range(1, len(close_v)):
        if dirs[t - 1] == 1 and not np.isnan(trail_v[t]) and close_v[t] < trail_v[t]:
            # a break below trail while previously long must flip aceDir to -1
            assert dirs[t] == -1


def test_simulate_exit_excludes_entries_not_long():
    close, high, low = _synthetic_trend()
    result = run_ace_white_line(high, low, close)
    forced_flat_pos = 300
    result.ace_dir.iloc[forced_flat_pos] = -1
    outcome = simulate_exit(result, close, entry_pos=forced_flat_pos)
    assert outcome.exit_reason == "excluded_ace_dir"
    assert outcome.exit_price is None


def test_simulate_exit_respects_max_hold():
    close, high, low = _synthetic_trend()
    result = run_ace_white_line(high, low, close)
    entry_pos = 260
    if result.ace_dir.iloc[entry_pos] != 1:
        return  # depends on synthetic path; skip if not a valid long entry
    outcome = simulate_exit(result, close, entry_pos=entry_pos, max_hold_days=60)
    assert outcome.exit_reason in ("trail_flip", "max_hold")
    assert outcome.exit_price is not None
