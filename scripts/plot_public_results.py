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
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/figures"
INK, MUTED, LINE = "#18273B", "#65758A", "#E5EBF3"
ACCENT, BASE, PALE = "#008D7F", "#9BAFCB", "#F0F4F8"
OFFLINE = "#3167B2"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 12, "text.color": INK,
    "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": LINE, "figure.facecolor": "white", "axes.facecolor": "white",
    "svg.fonttype": "none", "svg.hashsalt": "cam-public-results",
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
})


def rounded(value, decimals=1):
    return str(Decimal(str(value)).quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP))


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
    costs = json.loads((OUT / "benchmark_costs.json").read_text())["benchmarks"]
    fig = plt.figure(figsize=(17, 7.8))
    panels = {panel["benchmark"]: panel for panel in data["panels"]}
    excluded_methods = {("mikasa_robo", "MemoryVLA"), ("robomemarena", "FrameSamp+Modul")}
    for i, benchmark in enumerate(("rmbench", "robomme", "mikasa_robo", "robomemarena")):
        panel = panels[benchmark]
        left, width = .025 + i * .245, .215
        ax = fig.add_axes([left, .265, width, .65])
        rows = [row for row in panel["methods"]
                if (benchmark, row["method"]) not in excluded_methods]
        positions = np.arange(len(rows) - 1, -1, -1)
        colors = [BASE if row["method"] != "CaM" else
                  ACCENT if row["setting"].startswith("demo_interaction") else OFFLINE
                  for row in rows]
        ax.barh(positions, 100, height=.28, color=PALE, edgecolor="none", zorder=1)
        ax.barh(positions, [row["success_percent"] for row in rows], height=.28,
                color=colors, edgecolor="none", zorder=2)
        ax.set_xlim(0, 100)
        ax.set_ylim(-.35, len(rows) - .2)
        ax.set_title(panel["title"], loc="left", fontsize=25, weight="bold", pad=8)
        for y, row, color in zip(positions, rows, colors):
            text_color = INK if color == BASE else color
            weight = "normal" if color == BASE else "bold"
            method = "CaM w/ rollouts" if color == ACCENT else row["method"]
            ax.text(0, y + .23, method, va="bottom", fontsize=19,
                    color=text_color, weight=weight)
            label = rounded(row["success_percent"], row.get("display_decimals", 1)).rstrip("0").rstrip(".")
            ax.text(100, y + .23, label + "%", ha="right", va="bottom",
                    fontsize=19 if color == BASE else 21, color=text_color, weight=weight)
        ax.set_axis_off()

        # A compact, color-matched results key keeps the bar labels readable.
        key = fig.add_axes([left, .04, width, .19])
        key.set_xlim(0, 1)
        key.set_ylim(0, 1)
        key.plot([0, 1], [1, 1], color=LINE, lw=1.2)
        key.text(0, .82, "# Demos", fontsize=19, color=MUTED)
        key.text(1, .82, "API cost", ha="right", fontsize=19, color=MUTED)
        settings = [row for row in rows if row["method"] == "CaM"]
        for y, row in zip((.51, .10), settings):
            interaction = row["setting"].startswith("demo_interaction")
            color = ACCENT if interaction else OFFLINE
            cost = costs[panel["benchmark"]][row["setting"]]
            task_costs = {task["task"]: task for task in cost["tasks"]}
            for task, result in row["per_task"].items():
                audited = task_costs[task]["evaluation"]
                if any(result[k] != audited[k] for k in ("successes", "episodes")):
                    raise ValueError(f"Figure and audited evaluation disagree: {task}")
            if (row["successes"] != sum(t["successes"] for t in row["per_task"].values())
                or row["episodes"] != sum(t["episodes"] for t in row["per_task"].values())
                or not np.isclose(row["success_percent"], 100 * row["successes"] / row["episodes"])):
                raise ValueError("Figure totals do not match per-task results")
            demos = {task["demonstrations"] for task in row["per_task"].values()}
            budget = str(max(demos))
            key.scatter(.015, y + .03, s=100, marker="s", color=color, clip_on=False)
            key.text(.06, y, budget, fontsize=21, color=color, weight="bold", va="center")
            key.text(1, y, f'${cost["development_api_cost_usd"]:.2f}', ha="right",
                     va="center", fontsize=21, color=color, weight="bold")
        key.set_axis_off()
    save(fig, "benchmark_comparison")


def robomme_efficiency():
    data = json.loads((OUT / "robomme_comparison.json").read_text())
    rows = data["methods"]
    fig = plt.figure(figsize=(18, 6.4))
    positions = np.arange(len(rows) - 1, -1, -1)
    axes = [fig.add_axes([.03 + i * .33, .12, .285, .72]) for i in range(3)]
    titles = ["Success rate", "Policy computation", "API cost (USD)"]
    subtitles = ["Original RoboMME", "Mean ms / control step · log scale", "Development and evaluation separately"]
    for ax, title, subtitle in zip(axes, titles, subtitles):
        ax.set_ylim(-.5, len(rows) - .15)
        ax.set_title(title, loc="left", fontsize=24, weight="bold", pad=44)
        ax.text(0, 1.025, subtitle, transform=ax.transAxes, fontsize=14, color=MUTED)
        ax.set_axis_off()
    axes[0].set_xlim(0, 100)
    axes[1].set_xscale("log")
    axes[1].set_xlim(.1, 500)
    axes[2].set_xlim(0, 1)
    for row, y in zip(rows, positions):
        cam = row["id"] == "cam"
        color = ACCENT if cam else BASE
        weight = "bold" if cam else "normal"
        for ax in axes:
            ax.text(0, y + .27, row["method"], transform=ax.get_yaxis_transform(),
                    fontsize=18, weight=weight, color=ACCENT if cam else INK, va="bottom")
        ax = axes[0]
        ax.barh(y, 100, height=.28, color=PALE)
        ax.barh(y, row["success_percent"], height=.28, color=color)
        ax.text(100, y + .27, rounded(row["success_percent"]) + "%", ha="right", va="bottom",
                fontsize=18, color=ACCENT if cam else INK, weight=weight)
        ax = axes[1]
        value = row["policy_compute_ms_per_control_step"]
        label = "N/R" if value is None else f"{value:.2f}"
        upper = row.get("policy_compute_display_upper_bound_ms")
        if upper is not None:
            if value is None or not 0 <= value < upper:
                raise ValueError("Latency display bound does not contain the measured mean")
            label = f"<{upper:g} ms"
        ax.barh(y, 499.9, left=.1, height=.28, color=PALE)
        if value is not None:
            ax.barh(y, max(0, value - .1), left=.1, height=.28, color=color)
        ax.text(500, y + .27, label, ha="right", va="bottom",
                fontsize=18, color=ACCENT if cam else INK, weight=weight)
        ax = axes[2]
        dev, evaluation = row["development_api_cost_usd"], row["evaluation_api_cost_usd"]
        ax.text(.02, y - .02, "Develop", fontsize=13, color=MUTED, va="center")
        ax.text(.48, y - .02, "N/R" if dev is None else f"${dev:.2f}", ha="right", va="center",
                fontsize=18, color=ACCENT if cam else MUTED, weight=weight)
        ax.text(.55, y - .02, "Evaluate", fontsize=13, color=MUTED, va="center")
        ax.text(1, y - .02, "N/R" if evaluation is None else f"${evaluation:,.0f}", ha="right", va="center",
                fontsize=18, color=ACCENT if cam else INK, weight=weight)
        ax.plot([0, 1], [y - .30, y - .30], color=LINE, lw=.8)
    fig.text(.03, .047, "CaM timing: CPU replay, including action helpers. Baseline timing: RoboMME-Plus reference measurements.", fontsize=13, color=MUTED)
    fig.text(.03, .008, "API cost excludes local compute. N/R = not reported. Different hardware and evaluation protocols; see source notes.", fontsize=13, color=MUTED)
    save(fig, "robomme_efficiency")


if __name__ == "__main__":
    benchmark_comparison()
    robomme_efficiency()
