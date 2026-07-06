"""Deliverable generation — Section 6.

1. R-IVU01_report.md
2. R-IVU01_params.json  (W_VETO always null — F-13, commander decides post-hoc)
3. R-IVU01_weight.pine  (template selected by the Step-4 winning kernel, F-12)
4. data_quality_flags.csv is written directly by data/quality.py
"""

from __future__ import annotations

import json
from pathlib import Path

from r_ivu01.stats.gates import GateResult

PINE_TEMPLATE_LOG_GAUSSIAN = """\
rvol  = volume / ta.sma(volume, 20)[1]

z     = (math.log(math.max(rvol, 0.01)) - math.log({mu})) / {sigma}
w_vol = math.exp(-0.5 * z * z)

// 紅線：w_vol 僅供倉位縮放/CE評分；w_vol < W_VETO 維持硬否決
"""

PINE_TEMPLATE_GAMMA = """\
rvol  = volume / ta.sma(volume, 20)[1]

r     = math.max(rvol, 0.01) / {mu}
w_vol = math.pow(r, {k}) * math.exp({k} * (1.0 - r))

// 紅線：w_vol 僅供倉位縮放/CE評分；w_vol < W_VETO 維持硬否決
"""


def write_params_json(path: str, market: str, windows_results: dict) -> None:
    """windows_results: {h: {"family": "log_gaussian"|"gamma", "params": (mu, sigma_or_k)}}"""
    payload = {
        "market": market,
        "windows": {
            str(h): {
                "family": res["family"],
                "mu": res["params"][0],
                "sigma_or_k": res["params"][1],
                "W_VETO": None,  # F-13: commander decides post-acceptance-report, never derived here
                "weight_floor": None,
            }
            for h, res in windows_results.items()
        },
    }
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def write_pine_weight_file(path: str, kernel_name: str, params: tuple[float, float]) -> None:
    mu, sigma_or_k = params
    if kernel_name == "log_gaussian":
        content = PINE_TEMPLATE_LOG_GAUSSIAN.format(mu=mu, sigma=sigma_or_k)
    elif kernel_name == "gamma":
        content = PINE_TEMPLATE_GAMMA.format(mu=mu, k=sigma_or_k)
    else:
        raise ValueError(f"unknown kernel {kernel_name!r}, expected log_gaussian or gamma")
    Path(path).write_text(content)


def render_gate_table(gates: list[GateResult]) -> str:
    lines = ["| Gate | Passed | Detail |", "|---|---|---|"]
    for g in gates:
        lines.append(f"| {g.name} | {'PASS' if g.passed else 'FAIL'} | {g.detail} |")
    return "\n".join(lines)


def write_report_md(
    path: str,
    market: str,
    gates: list[GateResult],
    caveats: list[str],
) -> None:
    overall_pass = all(g.passed for g in gates if g.name != "G5_cross_market_consistency")
    lines = [
        "# R-IVU01 v1.3 — Falsification Report",
        "",
        f"Market: {market}",
        "",
        f"**Overall verdict: {'ACCEPT (proceed to commander review)' if overall_pass else 'FALSIFIED — archive'}**",
        "",
        "## Gates",
        "",
        render_gate_table(gates),
        "",
        "## Caveats (mandatory, per spec section 8)",
        "",
    ]
    lines.extend(f"- {c}" for c in caveats)
    Path(path).write_text("\n".join(lines))


MANDATORY_REPORT_CAVEATS = [
    "白線為結構線（最慢出場），實戰出場多由戰術線 trail_Main_L + Exit Escalation 先觸發；"
    "第二層實現R分佈代表「結構持有情境」，非生產出場分佈。此不影響 H4 檢定效力，但 G4 結論條件於此情境。",
    "OOS 期間（2023–2026）幾乎為單一多頭 regime，外推限制須標註（G2 對半切部分緩解）。",
]
