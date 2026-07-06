"""Entry point. This scaffold cannot run end-to-end without real data wiring
(finlab client for TW, corporate-action calendars) — see data/taiwan.py and
events/exclusions.py TODOs. `python -m r_ivu01.cli --market US ...` can run
further since yfinance is a public API, but the primary test combination
(spec E-06) is TW x h=5, so a TW data feed is required for any gate verdict.
"""

from __future__ import annotations

import argparse
import sys

from r_ivu01.config import PRIMARY_MARKET


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="R-IVU01 v1.3 pipeline runner")
    parser.add_argument("--market", choices=["TW", "US"], default=PRIMARY_MARKET)
    parser.add_argument("--out-dir", default="./out")
    parser.add_argument(
        "--resamples", type=int, default=1000, help="cluster bootstrap resamples for G1/G3"
    )
    args = parser.parse_args(argv)

    print(
        f"[r_ivu01] Scaffold only: wire data/{'taiwan' if args.market == 'TW' else 'us'}.py "
        "to a real feed before running the full pipeline (see module docstrings).",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
