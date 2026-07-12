import pandas as pd

from r_ivu01.data.quality import flag_extreme_moves


def test_only_extreme_moves_are_flagged():
    idx = pd.date_range("2020-01-01", periods=4, freq="B")
    close = pd.DataFrame(
        {
            "A": [10.0, 10.1, 20.0, 20.2],  # day 2: +98% -> flagged
            "B": [10.0, 10.1, 10.2, 10.3],  # no extreme moves
        },
        index=idx,
    )
    flagged = flag_extreme_moves(close)
    assert len(flagged) == 1
    row = flagged.iloc[0]
    assert row["ticker"] == "A"
    assert row["date"] == idx[2]
    assert row["pct_change"] > 0.5


def test_no_extreme_moves_returns_empty_frame():
    idx = pd.date_range("2020-01-01", periods=5, freq="B")
    close = pd.DataFrame({"A": [10.0, 10.1, 10.2, 10.1, 10.3]}, index=idx)
    flagged = flag_extreme_moves(close)
    assert len(flagged) == 0
    assert list(flagged.columns) == ["date", "ticker", "pct_change"]
