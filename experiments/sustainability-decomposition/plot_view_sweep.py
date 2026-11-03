"""Frontier chart for the shared-data-view sweep (runs/view_sweep/scores.json).

Small multiples, one column per technique, rows = combined score and activity accuracy. Each
panel shows that technique's levels as a connected line (one-change comparisons hollow), all
other plans in grey, the Pareto frontier over all plans as a dashed step line, and the
full-text pipeline as a reference. Writes runs/view_sweep/frontier.png.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs" / "view_sweep"
SURFACE, TEXT, TEXT_2, GRID, CONTEXT = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#c9c8c2"
SERIES = "#2a78d6"   # reference palette, categorical slot 1
TECHNIQUES = [("head_tail", "Head/tail"), ("topk", "Top-k retrieval"), ("jev", "Jev retrieval"),
              ("summarization", "Summarization")]
METRICS = [("combined", "Combined score"), ("activity", "Activity accuracy")]
# Full-text pipeline, same model, data and evaluator (runs/root, earlier phase 3).
FULL_TEXT = {"cost": 2.402, "combined": 0.81, "activity": 0.69}


def frontier(rows, key):
    best, out = -1.0, []
    for r in sorted(rows, key=lambda r: (r["cost"], -r[key])):
        if r[key] > best:
            out.append(r)
            best = r[key]
    return out


def is_level(plan):
    return plan.rsplit("_", 1)[-1].isdigit()


def main():
    rows = json.load(open(OUT / "scores.json"))
    fig, axes = plt.subplots(2, 4, figsize=(15, 7.4), sharex=True, facecolor=SURFACE)
    for row, (key, label) in enumerate(METRICS):
        front = frontier(rows, key)
        lo = min(r[key] for r in rows) - 0.02
        for col, (tech, title) in enumerate(TECHNIQUES):
            ax = axes[row][col]
            ax.set_facecolor(SURFACE)
            others = [r for r in rows if r["technique"] != tech]
            ax.scatter([r["cost"] for r in others], [r[key] for r in others], s=22, color=CONTEXT, zorder=1)
            xs, ys = [], []
            for i, r in enumerate(front):
                nxt = front[i + 1]["cost"] if i + 1 < len(front) else 3.0
                xs += [r["cost"], nxt]
                ys += [r[key], r[key]]
            ax.plot(xs, ys, color=TEXT_2, lw=1.2, ls=(0, (4, 3)), zorder=2)
            ax.scatter([FULL_TEXT["cost"]], [FULL_TEXT[key]], marker="D", s=40, color=TEXT_2, zorder=3)
            mine = sorted([r for r in rows if r["technique"] == tech and is_level(r["plan"])],
                          key=lambda r: int(r["plan"].rsplit("_", 1)[-1]))
            ax.plot([r["cost"] for r in mine], [r[key] for r in mine], color=SERIES, lw=2, zorder=4)
            ax.scatter([r["cost"] for r in mine], [r[key] for r in mine], s=64, color=SERIES,
                       edgecolor=SURFACE, linewidth=2, zorder=5)
            for r in mine:
                ax.annotate(r["plan"].rsplit("_", 1)[-1], (r["cost"], r[key]), textcoords="offset points",
                            xytext=(0, 8), ha="center", fontsize=8, color=TEXT)
            for r in [r for r in rows if r["technique"] == tech and not is_level(r["plan"])]:
                ax.scatter([r["cost"]], [r[key]], s=64, facecolor=SURFACE, edgecolor=SERIES, linewidth=2, zorder=5)
                ax.annotate(r["plan"].split("_", 2)[-1].replace("3_", "").replace("_", " "),
                            (r["cost"], r[key]), textcoords="offset points", xytext=(6, -10),
                            fontsize=7.5, color=TEXT_2)
            ax.set_xscale("log")
            ax.set_xlim(0.03, 3.0)
            ax.set_ylim(lo, 0.86 if key == "combined" else 0.74)
            ax.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5, 1, 2]))
            ax.xaxis.set_minor_locator(NullLocator())
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
            ax.grid(axis="y", color=GRID, lw=0.8)
            ax.set_axisbelow(True)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID)
            ax.tick_params(colors=TEXT_2, labelsize=8, length=0)
            if row == 0:
                ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=8)
            if col == 0:
                ax.set_ylabel(label, fontsize=9, color=TEXT_2)
            if row == 1:
                ax.set_xlabel("Cost for 50 reports (USD, log)", fontsize=8.5, color=TEXT_2)

    handles = [
        plt.Line2D([], [], color=SERIES, lw=2, marker="o", markersize=7, markeredgecolor=SURFACE,
                   label="This technique: levels 1-5 (less to more text)"),
        plt.Line2D([], [], ls="", marker="o", markersize=7, markerfacecolor=SURFACE, markeredgecolor=SERIES,
                   markeredgewidth=2, label="One-change comparison"),
        plt.Line2D([], [], ls="", marker="o", markersize=5, color=CONTEXT, label="Other techniques"),
        plt.Line2D([], [], color=TEXT_2, lw=1.2, ls=(0, (4, 3)), label="Pareto frontier (all plans)"),
        plt.Line2D([], [], ls="", marker="D", markersize=6, color=TEXT_2, label="Full text, no decomposition"),
    ]
    fig.legend(handles=handles, loc="upper left", ncol=5, frameon=False, fontsize=8.5,
               bbox_to_anchor=(0.04, 0.995), labelcolor=TEXT)
    fig.text(0.04, 0.01, "Sustainability pipeline (filter -> classify -> summary reduce) on train_50, all LLM steps on "
             "gemini-3.5-flash-lite, one shared data view per plan; corrected evaluator; one run each "
             "(1 report = about 0.023).", fontsize=8, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    fig.savefig(OUT / "frontier.png", dpi=150, facecolor=SURFACE)
    for key, _ in METRICS:
        print(f"frontier ({key}):", " -> ".join(f"{r['plan']} (${r['cost']:.3f}, {r[key]:.2f})" for r in frontier(rows, key)))
    print("wrote", OUT / "frontier.png")


if __name__ == "__main__":
    main()
