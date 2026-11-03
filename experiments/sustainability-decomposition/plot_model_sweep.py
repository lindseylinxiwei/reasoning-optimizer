"""Pareto frontier for the model sweep on the frontier plan (head 500 tokens).

Reads runs/model_sweep/scores.json, writes runs/model_sweep/frontier_models.png. Same styling as
plot_all_nodes.py: frontier in blue, every other plan grey, full-text pipeline as a reference.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

from plot_all_nodes import FRONT, FULL_TEXT, GRID, POINT, SURFACE, TEXT, TEXT_2, frontier  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs" / "model_sweep"


def label(plan):
    return "Jev (filter + classify)" if plan == "h1_jev" else plan.removeprefix("h1_")


def main():
    rows = json.load(open(OUT / "scores.json"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, [("combined", "Combined score (activity + names)"),
                                       ("activity", "Activity accuracy (MOAR's search reward)")]):
        ax.set_facecolor(SURFACE)
        front = frontier(rows, key)
        on = {r["plan"] for r in front}
        xs, ys = [], []
        for i, r in enumerate(front):
            nxt = front[i + 1]["cost"] if i + 1 < len(front) else 3.2
            xs += [r["cost"], nxt]
            ys += [r[key], r[key]]
        ax.plot(xs, ys, color=FRONT, lw=1.5, ls=(0, (4, 3)), zorder=1)
        for r in rows:
            marker = "^" if r["plan"] == "h1_jev" else "o"
            ax.scatter(r["cost"], r[key], marker=marker, s=90 if r["plan"] in on else 60,
                       color=FRONT if r["plan"] in on else POINT, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        # 3.6/3.7/3.8-flash land on nearly the same point: spread their labels left, above, right.
        offsets = {"h1_3.7-flash": (-8, 0), "h1_3.8-flash": (0, 9), "h1_3.6-flash": (8, 0)}
        for r in rows:
            dx, dy = offsets.get(r["plan"], (0, 9))
            ha = "right" if dx < 0 else "left" if dx > 0 else "center"
            ax.annotate(label(r["plan"]), (r["cost"], r[key]), textcoords="offset points", xytext=(dx, dy),
                        ha=ha, va="bottom" if dy > 0 else "center", fontsize=10.5, color=TEXT)
        ax.scatter(FULL_TEXT["cost"], FULL_TEXT[key], marker="*", s=150, color=TEXT, zorder=3)
        ax.annotate("full text, 3.5-flash-lite", (FULL_TEXT["cost"], FULL_TEXT[key]), textcoords="offset points",
                    xytext=(0, -12), ha="center", va="top", fontsize=11, color=TEXT)
        ax.set_xscale("log")
        ax.set_xlim(0.012, 3.2)
        ax.xaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
        ax.set_title(title, loc="left", fontsize=14, color=TEXT, pad=10)
        ax.set_xlabel("Cost for 50 reports (USD, log scale)", fontsize=12, color=TEXT_2)
        ax.set_ylabel("Score", fontsize=12, color=TEXT_2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=TEXT_2, labelsize=11.5, length=0)
    handles = [plt.Line2D([], [], ls="", marker="o", markersize=7, color=POINT, label="All three LLM steps on one model"),
               plt.Line2D([], [], ls="", marker="^", markersize=8, color=POINT, label="Jev for filter + classify"),
               plt.Line2D([], [], ls="", marker="o", markersize=8, color=FRONT, label="On the Pareto frontier"),
               plt.Line2D([], [], color=FRONT, lw=1.5, ls=(0, (4, 3)), label="Pareto frontier"),
               plt.Line2D([], [], ls="", marker="*", markersize=10, color=TEXT, label="Full text (no decomposition)")]
    fig.legend(handles=handles, loc="upper left", ncol=5, frameon=False, fontsize=11.5,
               bbox_to_anchor=(0.04, 0.995), labelcolor=TEXT)
    fig.text(0.04, 0.012, "Every plan uses the head-500-token view (H1). Model names drop the gemini- prefix. train_50, "
             "corrected evaluator, one run each (1 report = about 0.023).", fontsize=11, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.04, 1, 0.92))
    fig.savefig(OUT / "frontier_models.png", dpi=110, facecolor=SURFACE)
    print("wrote", OUT / "frontier_models.png")


if __name__ == "__main__":
    main()
