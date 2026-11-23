"""Figures for the README, from results/*.json.

  python scripts/make_figures.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

R = Path(__file__).resolve().parents[1] / "results"
OUT = Path(__file__).resolve().parents[1] / "docs"
OUT.mkdir(exist_ok=True)

BASE_C, GRAFT_C, SIM_C = "#9aa0a6", "#1a73e8", "#34a853"


def digits_of(prompt):
    nums = re.findall(r"\d+", prompt)
    return max(len(n) for n in nums[-2:]) if nums else 0


def fig_accuracy():
    """Same 600 prompts, same greedy decoding, both models (scripts/continuation_eval.py)."""
    g = json.loads((R / "continuation_graft.json").read_text())
    b = json.loads((R / "continuation_base.json").read_text())
    ks = list(range(1, 7))
    gy = [100 * g["by_digits"][str(k)] for k in ks]
    by = [100 * b["by_digits"][str(k)] for k in ks]
    fig, ax = plt.subplots(figsize=(6.4, 3.6), dpi=150)
    ax.plot(ks, by, "o-", color=BASE_C, lw=2, label="Qwen3-0.6B")
    ax.plot(ks, gy, "o-", color=GRAFT_C, lw=2.5, label="Qwen3-0.6B + grafted circuit (0 gradients)")
    ax.set_xlabel("digits in the longer operand  (100 prompts each, A+B= and A-B=)")
    ax.set_ylabel("exact answers, greedy (%)")
    ax.set_ylim(-3, 108)
    ax.set_xticks(ks)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "accuracy.png")
    print("wrote", OUT / "accuracy.png", gy, by)


def fig_robustness():
    pts = []
    for f in sorted(R.glob("sim_spread_*.json")):
        d = json.loads(f.read_text())
        pts.append((d["spread"], 100 * d["exact"]))
    if not pts:
        return
    pts.sort()
    fig, ax = plt.subplots(figsize=(6.4, 3.2), dpi=150)
    ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", color=SIM_C, lw=2.5)
    ax.axvline(1.0, color=BASE_C, ls="--", lw=1)
    ax.text(1.02, 8, "range seen in\ncalibration", color="#5f6368", fontsize=8)
    ax.set_xlabel("RMSNorm scale range, as a multiple of the calibrated log-range")
    ax.set_ylabel("exact (%)  [circuit simulator]")
    ax.set_ylim(-3, 103)
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "robustness.png")
    print("wrote", OUT / "robustness.png", pts)


if __name__ == "__main__":
    fig_accuracy()
    fig_robustness()
