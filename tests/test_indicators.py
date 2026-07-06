import numpy as np
import pandas as pd
import pytest

from r_ivu01.indicators.atr import atr, wilder_rma
from r_ivu01.indicators.rvol import rvol
from r_ivu01.indicators.forward import forward_indicators


def test_wilder_rma_seed_is_sma():
    s = pd.Series([1, 2, 3, 4, 5, 6, 7, 8])
    out = wilder_rma(s, period=4)
    assert np.isnan(out.iloc[2])
    assert out.iloc[3] == pytest.approx(2.5)
    # recursive step: rma[4] = rma[3] + (x[4]-rma[3])/4
    assert out.iloc[4] == pytest.approx(2.5 + (5 - 2.5) / 4)


def test_atr_non_negative():
    idx = pd.date_range("2020-01-01", periods=30, freq="B")
    close = pd.Series(100 + np.cumsum(np.random.default_rng(0).normal(0, 1, 30)), index=idx)
    high = close + 1
    low = close - 1
    result = atr(high, low, close, period=14)
    valid = result.dropna()
    assert (valid >= 0).all()


def test_rvol_uses_prior_window_only():
    idx = pd.date_range("2020-01-01", periods=25, freq="B")
    volume = pd.Series([100.0] * 20 + [1000.0] * 5, index=idx)
    r = rvol(volume, window=20)
    # event day itself (a big-volume day) must not appear in its own denominator
    assert r.iloc[20] == pytest.approx(1000.0 / 100.0)


def test_forward_indicators_denominator_is_t_minus_1():
    idx = pd.date_range("2020-01-01", periods=40, freq="B")
    rng = np.random.default_rng(1)
    close = pd.Series(100 + np.cumsum(rng.normal(0, 1, 40)), index=idx)
    high, low = close + 0.5, close - 0.5
    fwd = forward_indicators(close, high, low, atr_period=14, windows=(5,))
    atr_direct = atr(high, low, close, 14).shift(1)
    pd.testing.assert_series_equal(fwd["atr_t_minus_1"], atr_direct, check_names=False)
