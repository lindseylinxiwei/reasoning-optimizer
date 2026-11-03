"""Test-set (100 reports) frontier: our two frontier plans against MOAR's frontier nodes.

MOAR nodes are its search-time frontier (nodes 15, 17, 20 on gemini-2.5-flash-lite, node 20 with
gpt-5-nano steps), re-scored from their saved test outputs with the corrected evaluator. Node 13 is
left out because its plan read ground-truth labels during the search. Our plans come from
runs/model_sweep_test (view_sweep.py::model_main --split test). Writes
runs/model_sweep_test/frontier_test.png and compare.json.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

from plot_all_nodes import FRONT, GRID, POINT, SURFACE, TEXT, TEXT_2, frontier  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO), str(REPO / "experiments/reasoning")]
from evaluation.sustainability import evaluate_results  # noqa: E402

OUT = HERE / "runs" / "model_sweep_test"
GT = REPO / "experiments/reasoning/data/company_reports_gt.json"
TEST = REPO / "experiments/reasoning/data/test/sustainability.json"
MOAR_LOGS = Path.home() / "Downloads/search_logs/sustainability"
MOAR_NODES = {15: "MOAR 15 (2.5-flash-lite)", 17: "MOAR 17 (2.5-flash-lite)", 20: "MOAR 20 (2.5-flash-lite + gpt-5-nano)"}
OURS = {"h1_jev": "Head 500 + Jev", "h1_3.1-flash-lite": "Head 500, 3.1-flash-lite"}


def scores(output):
    m = evaluate_results("test", str(output), str(GT), str(TEST))
    return {"combined": m["combined_score"], "activity": m["economic_activity_accuracy"],
            "names": m["company_name_accuracy"], "found": m["target_reports"] - m["missing_companies"],
            "targets": m["target_reports"], "let_through": m["non_target_reports_in_output"],
            "non_targets": m["non_target_reports"]}


def rows():
    out = []
    moar = {p["node_id"]: p for p in json.load(open(MOAR_LOGS / "pareto_frontier_sustainability.json"))["frontier_points"]}
    for node, name in MOAR_NODES.items():
        p = moar[node]
        out.append({"plan": name, "source": "moar", "cost": p["test_cost"],
                    **scores(MOAR_LOGS / "test_plans/mcts" / p["file"])})
    for plan, name in OURS.items():
        meta = json.load(open(OUT / plan / "run_meta.json"))
        out.append({"plan": name, "source": "ours", "cost": meta["cost"], **scores(OUT / plan / "output.json")})
    return out


def main():
    data = rows()
    json.dump(data, open(OUT / "compare.json", "w"), indent=1)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), facecolor=SURFACE)
    for ax, (key, title) in zip(axes, [("combined", "Combined score (activity + names)"),
                                       ("activity", "Activity accuracy (MOAR's search reward)")]):
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
            marker = "s" if r["source"] == "moar" else "o"
            ax.scatter(r["cost"], r[key], marker=marker, s=110 if r["plan"] in on else 80,
                       color=FRONT if r["plan"] in on else POINT, edgecolor=SURFACE, linewidth=1.5, zorder=3)
            below = r["plan"] in ("Head 500, 3.1-flash-lite", "MOAR 17 (2.5-flash-lite)")
            ax.annotate(r["plan"], (r["cost"], r[key]), textcoords="offset points", xytext=(0, -12 if below else 10),
                        ha="center", va="top" if below else "bottom", fontsize=10.5, color=TEXT)
        ax.set_xscale("log")
        ax.set_xlim(0.015, 1.5)
        lo = min(r[key] for r in data)
        ax.set_ylim(lo - 0.03, max(r[key] for r in data) + 0.03)
        ax.xaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.2, 0.5, 1]))
        ax.xaxis.set_minor_locator(NullLocator())
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
        ax.set_title(title, loc="left", fontsize=14, color=TEXT, pad=10)
        ax.set_xlabel("Cost for 100 test reports (USD, log scale)", fontsize=12, color=TEXT_2)
        ax.set_ylabel("Score", fontsize=12, color=TEXT_2)
        ax.grid(axis="y", color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=TEXT_2, labelsize=11.5, length=0)
    handles = [plt.Line2D([], [], ls="", marker="s", markersize=8, color=POINT, label="MOAR frontier node"),
               plt.Line2D([], [], ls="", marker="o", markersize=8, color=POINT, label="Our frontier plan"),
               plt.Line2D([], [], ls="", marker="o", markersize=9, color=FRONT, label="On the test Pareto frontier"),
               plt.Line2D([], [], color=FRONT, lw=1.5, ls=(0, (4, 3)), label="Pareto frontier")]
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=11.5,
               bbox_to_anchor=(0.04, 0.995), labelcolor=TEXT)
    fig.text(0.04, 0.008, "100-report test set (82 sustainability reports), corrected evaluator, one run each "
             "(1 report = about 0.011).\nMOAR node 13 is left out: its plan read the ground-truth labels during search.",
             fontsize=11, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.93))
    fig.savefig(OUT / "frontier_test.png", dpi=110, facecolor=SURFACE)
    for r in sorted(data, key=lambda r: r["cost"]):
        print(f"{r['plan']:40s} ${r['cost']:.4f} combined {r['combined']:.3f} activity {r['activity']:.3f} "
              f"names {r['names']:.3f} found {r['found']}/{r['targets']} let through {r['let_through']}/{r['non_targets']}")
    print("wrote", OUT / "frontier_test.png")


if __name__ == "__main__":
    main()
