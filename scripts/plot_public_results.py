"""Render publication figures from the committed, sourced measurements.

Run: python scripts/plot_public_results.py
Writes PNG, SVG and PDF to docs/figures. No experiment or network calls.
"""
from decimal import Decimal, ROUND_HALF_UP
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"
INK, MUTED, LINE = "#18273B", "#65758A", "#E5EBF3"
ACCENT, BASE, PALE = "#008D7F", "#9BAFCB", "#F0F4F8"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": LINE, "figure.facecolor": "white", "axes.facecolor": "white",
    "svg.fonttype": "none", "svg.hashsalt": "cam-public-results",
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
})


def rounded(value):
    return str(Decimal(str(value)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))


def save(fig, name):
    for suffix in ("png", "svg", "pdf"):
        path = OUT / f"{name}.{suffix}"
        metadata = None
        if suffix == "pdf":
            metadata = {"Creator": "CaM / scripts/plot_public_results.py", "CreationDate": None, "ModDate": None}
        elif suffix == "svg":
            metadata = {"Creator": "CaM / scripts/plot_public_results.py", "Date": None}
        fig.savefig(path, dpi=220, facecolor="white", metadata=metadata)
        if suffix == "svg":
            path.write_text("\n".join(line.rstrip() for line in path.read_text().splitlines()) + "\n")
    plt.close(fig)


def benchmark_comparison():
    data = json.loads((OUT / "benchmark_results.json").read_text())
    fig = plt.figure(figsize=(17, 6.4))
    fig.text(.035, .94, "Robot Memory as Code", fontsize=25, weight="bold")
    fig.text(.035, .884, "Success across four benchmarks", fontsize=15, color=MUTED)
    fig.text(.965, .934, "CaM", color=ACCENT, fontsize=24, weight="bold", ha="right")
    details = ["16 tasks · 800 CaM episodes", "9 shared tasks · 900 CaM episodes",
               "5 shared tasks · 250 CaM episodes", "26 tasks · task success rate (TSR)"]
    for i, (panel, detail) in enumerate(zip(data["panels"], details)):
        ax = fig.add_axes([.035 + i * .245, .17, .205, .59])
        ax.set_xlim(0, 100)
        ax.set_ylim(-.5, 5.7)
        ax.set_title(panel["title"], loc="left", fontsize=17, weight="bold", pad=27)
        ax.text(0, 1.025, detail, transform=ax.transAxes, fontsize=9.5, color=MUTED)
        for y, row in zip(np.linspace(5, 0, len(panel["methods"])), panel["methods"]):
            cam = row["method"] == "CaM"
            color = ACCENT if cam else BASE
            ax.barh(y, 100, height=.29, color=PALE, edgecolor="none", zorder=1)
            ax.barh(y, row["success_percent"], height=.29, color=color, edgecolor="none", zorder=2)
            ax.text(0, y + .27, row["method"], fontsize=12, weight="bold" if cam else "normal",
                    color=ACCENT if cam else INK, va="bottom")
            ax.text(100, y + .27, rounded(row["success_percent"]) + "%", ha="right", va="bottom",
                    fontsize=12, color=ACCENT if cam else INK, weight="bold" if cam else "normal")
        ax.set_yticks([])
        ax.set_xticks([0, 50, 100])
        ax.tick_params(axis="x", length=0, pad=9, labelsize=10)
        ax.spines["left"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.set_xlabel("Success (%)", fontsize=10, labelpad=9)
    fig.text(.035, .052, "Baselines are published reports; training, sensing, interaction budgets and evaluation protocols differ.",
             fontsize=10, color=MUTED)
    fig.text(.965, .052, "Sources & full results: docs/results.md", fontsize=10, color=MUTED, ha="right")
    save(fig, "benchmark_comparison")


def robomme_efficiency():
    data = json.loads((OUT / "robomme_efficiency.json").read_text())
    fig = plt.figure(figsize=(15, 6.8))
    fig.text(.055, .935, "RoboMME · memory decisions and API spend", fontsize=24, weight="bold")
    fig.text(.055, .878, "16 frozen programs · 25 demonstrations per task · no evaluation-time model calls", fontsize=13, color=MUTED)

    # The entire distribution makes expensive visual decisions visible alongside
    # fast memory updates. Each point is the median of five warm measurements.
    values = np.sort([row["median_ms"] for row in data["decisions_data"]])
    ax = fig.add_axes([.075, .21, .43, .49])
    ax.step(values, np.arange(1, len(values) + 1) / len(values) * 100,
            where="post", lw=2.8, color=ACCENT)
    ax.fill_between(values, 0, np.arange(1, len(values) + 1) / len(values) * 100,
                    step="post", alpha=.08, color=ACCENT)
    ax.set_xscale("log")
    ax.set_xlim(.02, max(1000, float(values[-1]) * 1.15))
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xticks([.1, 1, 10, 100, 1000])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.set_xlabel("Perception + memory decision time (ms, log scale)", fontsize=11, labelpad=10)
    ax.set_ylabel("Recorded decisions (%)", fontsize=11, labelpad=10)
    ax.grid(axis="y", color=LINE, lw=.8)
    ax.set_axisbelow(True)
    ax.axvline(data["median_ms"], ls=(0, (3, 4)), color=ACCENT, alpha=.65, lw=1)
    ax.axvline(data["p95_ms"], ls=(0, (3, 4)), color=MUTED, alpha=.65, lw=1)
    fig.text(.075, .768, f"{data['median_ms']:.2f} ms", fontsize=24, color=ACCENT, weight="bold")
    fig.text(.075, .724, "MEDIAN", fontsize=9, color=MUTED)
    fig.text(.272, .768, f"{data['p95_ms']:.1f} ms", fontsize=20, weight="bold")
    fig.text(.272, .724, "95TH PERCENTILE", fontsize=9, color=MUTED)
    fig.text(.505, .768, f"{data['mean_ms']:.1f} ms", fontsize=15, ha="right")
    fig.text(.505, .724, "MEAN", fontsize=9, color=MUTED, ha="right")

    cost = fig.add_axes([.60, .25, .34, .40])
    cost.barh([1, 0], [data["development_api_cost_usd"], 0], height=.32,
              color=[ACCENT, BASE], zorder=2)
    cost.scatter([0], [0], s=65, color=ACCENT, zorder=3, clip_on=False)
    cost.set_xlim(0, 35)
    cost.set_ylim(-.5, 1.55)
    cost.set_yticks([])
    cost.set_xticks([0, 10, 20, 30])
    cost.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    cost.grid(axis="x", color=LINE, lw=.8)
    cost.set_axisbelow(True)
    cost.spines["left"].set_visible(False)
    cost.set_xlabel("Reported API cost (USD)", fontsize=11, labelpad=11)
    cost.text(0, 1.27, "Development · all 16 tasks", fontsize=12, weight="bold")
    cost.text(0, .27, "Evaluation · 800 episodes", fontsize=12, weight="bold")
    cost.text(data["development_api_cost_usd"] + 1, 1, f"${data['development_api_cost_usd']:.2f}", va="center", fontsize=12, color=ACCENT, weight="bold")
    cost.text(1, 0, "$0.00", va="center", fontsize=12, color=ACCENT, weight="bold")
    fig.text(.60, .768, f"${data['development_api_cost_usd']:.2f}", fontsize=30, color=ACCENT, weight="bold")
    fig.text(.60, .714, "TOTAL API SPEND", fontsize=10, color=MUTED)
    fig.text(.94, .77, "795 / 800", fontsize=19, weight="bold", ha="right")
    fig.text(.94, .726, "HELD-OUT SUCCESSES", fontsize=9, color=MUTED, ha="right")
    fig.text(.055, .095, f"Timing: {data['decisions']} decisions from {data['episodes']} recorded episodes; 5 repeats per decision on CPU.", fontsize=10, color=MUTED)
    fig.text(.055, .055, "Decision timing excludes act tools, I/O and simulation. API estimates exclude local compute and data preparation.", fontsize=10, color=MUTED)
    save(fig, "robomme_efficiency")


if __name__ == "__main__":
    benchmark_comparison()
    robomme_efficiency()
