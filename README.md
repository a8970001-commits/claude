# R-IVU01 v1.3 — RVOL Inverted-U Volume Weighting Research Factor

Code scaffold for the spec `R-IVU01_研究規格書_v1.3` (EXECUTABLE status,
2026-07-06). This is a research pipeline, not a trading system — it exists to
falsify or corroborate H1 (inverted-U shape of forward risk-adjusted returns
vs. RVOL), not to produce a signal.

## What's implemented vs. what needs wiring

Fully implemented (pure logic, spec-locked constants, unit-tested):

- `r_ivu01/config.py` — every locked numeric constant in the spec (sample
  windows, exclusion rules, ATR/RVOL windows, ACE parameters, gate
  thresholds, the corrected G4 fee structure).
- `r_ivu01/indicators/atr.py` — Wilder RMA ATR (SMA-based ATR is banned).
- `r_ivu01/indicators/rvol.py` — RVOL with the `t-1` denominator.
- `r_ivu01/indicators/forward.py` — layer-1 `R_fwd[h]` / `win[h]`, with the
  F-01 fix (denominator = `ATR(14)[t-1]`, not `ATR(14)[t]`).
- `r_ivu01/indicators/ace.py` — the ACE white line (`trail_Struct_L`) state
  machine, transcribed verbatim from the frozen v5.25 text embedded in spec
  section 3.3 (asymmetric `base_atr_regime`, dynamic multiplier/lookback,
  ratchet + flip + rearm), plus the P2 Option-A exit simulator.
- `r_ivu01/events/breakout.py` — the 3.1 breakout definition + 10-day dedup.
- `r_ivu01/events/exclusions.py` — the 2.3/F-08 exclusion rules, as pure
  functions over pre-supplied corporate-action dates (see below).
- `r_ivu01/stats/` — decile/anchor-bin binning (F-04), ISO-week cluster
  bootstrap (E-05), the F-02 rewritten shape gate (peak-internality +
  dual-wing floor, replacing the fail-open nested-fit gate), WLS log-Gaussian
  vs. Gamma kernel selection (F-11), and gates G1-G5.
- `r_ivu01/pipeline.py` — orchestrates Steps 1-6 end-to-end; verified against
  synthetic random-walk data (see `tests/`) — a real inverted-U shape is
  correctly detected, and pure noise correctly fails G1.

Needs real data to run for real (deliberately stubbed, not faked):

- `r_ivu01/data/taiwan.py` — `load_taiwan_universe()` raises
  `NotImplementedError`; wire it to the commander's finlab client (裁決A).
  The docstring lists the exact required fields.
- `r_ivu01/data/us.py` — implemented against the real yfinance API (裁決B,
  all three 軍規: explicit `auto_adjust=True`, extreme-move flagging,
  survivorship-bias tagging) — this one can actually run once network access
  to Yahoo Finance is available.
- Corporate-action calendars (ex-dividend, rights issue/split/capital
  reduction, TW disposition flags, US split dates) — `events/exclusions.py`
  takes these as plain `DatetimeIndex` / boolean-Series arguments; nothing
  in the exclusion logic itself is missing, only the upstream feed.
- The F-14 TradingView replication spot-check (10 tickers x 3 events,
  ±2% tolerance) is a manual/semi-manual comparison step, not automatable
  without TradingView access.
- `W_VETO` is intentionally never computed anywhere in this code (F-13) —
  it's a post-report commander decision, not a pipeline output.

## Layout

```
r_ivu01/
  config.py          locked constants
  data/               finlab (stub) / yfinance loaders, quality flagging
  events/             breakout detection, 2.3/F-08 exclusion rules
  indicators/         ATR, RVOL, layer-1 forward returns, ACE white line
  stats/              binning, cluster bootstrap, shape gate, kernel fit, gates
  pipeline.py         Steps 1-6 orchestration
  report.py           Section 6 deliverables (report.md, params.json, .pine)
  cli.py              entry point (fails loudly until data/taiwan.py is wired)
tests/                unit tests for the pure-logic pieces + a pipeline smoke test
```

## Running tests

```
pip install -r requirements.txt
pytest tests/ -q
```

## Hard constraints carried over from the spec (do not violate in code)

- No TW/US parameter sharing or pooled-sample fitting.
- No hysteresis/smoothing/extra free parameters beyond what's specified.
- No reconstructing the ACE white line from first principles — only the
  frozen v5.25 text in `indicators/ace.py`'s docstring is authoritative.
- Any gate failure is reported as-is; do not loosen thresholds or redefine
  Section 3 to force a pass — that requires a new spec version.
