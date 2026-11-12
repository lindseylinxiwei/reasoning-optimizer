"""Test-set (100 reports) frontier: the sustainability_moar_500 frontier plans (#64, #39) against the
original MOAR frontier nodes 15/17/20 (from runs/model_sweep_test/compare.json, corrected evaluator). The new plans' test outputs come from
run_moar500_test.py. Usage: python plot_moar500_test.py <test run dir> <output dir>
Writes test_frontier.png and test_compare.json to the output dir.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

from plot_all_nodes import FRONT, GRID, POINT, SURFACE, TEXT, TEXT_2, frontier  # noqa: E402
from plot_test_frontier import OUT as OLD, scores  # noqa: E402

NEW = {"moar500_64": "#64 (3-flash-preview + Jev)", "moar500_39": "#39 (argon)"}
OLD_NAMES = {"MOAR 15 (2.5-flash-lite)": "MOAR 15", "MOAR 17 (2.5-flash-lite)": "MOAR 17",
             "MOAR 20 (2.5-flash-lite + gpt-5-nano)": "MOAR 20"}
MARKERS = {"moar": "s", "moar500": "D"}
LABELS = {"moar": "Original MOAR", "moar500": "New search"}
# label placement (dx, dy in points, ha), per panel, so names don't overlap
PLACE = {"combined": {"MOAR 15": (0, -12, "center"), "MOAR 17": (0, -12, "center"), "MOAR 20": (0, -12, "center"),
                      NEW["moar500_64"]: (0, 10, "center"), NEW["moar500_39"]: (0, 10, "center")},
         "activity": {"MOAR 15": (0, -12, "center"), "MOAR 17": (0, -12, "center"), "MOAR 20": (10, 0, "left"),
                      NEW["moar500_64"]: (0, 10, "center"), NEW["moar500_39"]: (0, 10, "center")}}


def main(test_dir: Path, out_dir: Path):
    data = [{**r, "plan": OLD_NAMES[r["plan"]]} for r in json.load(open(OLD / "compare.json")) if r["plan"] in OLD_NAMES]
    for plan, name in NEW.items():
        meta = json.load(open(test_dir / plan / "run_meta.json"))
        data.append({"plan": name, "source": "moar500", "cost": meta["cost"], **scores(test_dir / plan / "output.json")})
    out_dir.mkdir(parents=True, exist_ok=True)
    json.dump(data, open(out_dir / "test_compare.json", "w"), indent=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, [("combined", "Combined score"), ("activity", "Activity accuracy")]):
        ax.set_facecolor(SURFACE)
        front = frontier(data, key)
        on = {r["plan"] for r in front}
        xs, ys = [], []
        for i, r in enumerate(front):
            nxt = front[i + 1]["cost"] if i + 1 < len(front) else 1.5
            xs += [r["cost"], nxt]
            ys += [r[key], r[key]]
        ax.plot(xs, ys, color=FRONT, lw=1.5, ls=(0, (4, 3)), zorder=1)
        for r in data:
            ax.scatter(r["cost"], r[key], marker=MARKERS[r["source"]], s=120 if r["plan"] in on else 85,
                       color=FRONT if r["plan"] in on else POINT, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            dx, dy, ha = PLACE[key].get(r["plan"], (0, 10, "center"))
            ax.annotate(r["plan"], (r["cost"], r[key]), textcoords="offset points", xytext=(dx, dy), ha=ha,
                        va="bottom" if dy > 0 else "top" if dy < 0 else "center", fontsize=11, color=TEXT)
        ax.set_xscale("log")
        ax.set_xlim(0.015, 1.5)
        ax.set_ylim(min(r[key] for r in data) - 0.03, max(r[key] for r in data) + 0.03)
        ax.xaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.2, 0.5, 1]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
        ax.set_title(title, loc="left", fontsize=14, color=TEXT, pad=10)
        ax.set_xlabel("Cost on test set (log scale)", fontsize=12, color=TEXT_2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=TEXT_2, labelsize=11.5, length=0)
    handles = [plt.Line2D([], [], ls="", marker=MARKERS[s], markersize=8, color=POINT, label=LABELS[s])
               for s in ("moar", "moar500")]
    handles += [plt.Line2D([], [], color=FRONT, lw=1.5, ls=(0, (4, 3)), marker="o", markersize=7,
                           label="Pareto frontier")]
    fig.legend(handles=handles, loc="upper left", ncol=3, frameon=False, fontsize=11.5,
               bbox_to_anchor=(0.03, 0.995), labelcolor=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_dir / "test_frontier.png", dpi=110, facecolor=SURFACE)
    print("wrote", out_dir / "test_frontier.png", out_dir / "test_compare.json")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]))
