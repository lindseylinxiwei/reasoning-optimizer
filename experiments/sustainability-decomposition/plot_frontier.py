"""Rescore saved phase 3 outputs with the corrected evaluator and plot cost vs. accuracy.

All runs here use gemini-3.5-flash-lite on train_50 (50 reports). Two panels: activity
accuracy (MOAR's search reward) and combined score (used for the reported frontier).
Writes runs/frontier.png and runs/frontier_scores.json.

Usage (from the repository root):
  python experiments/sustainability-decomposition/plot_frontier.py
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "experiments/reasoning"))
from evaluation.sustainability import evaluate_results  # noqa: E402

RUNS = HERE / "runs"
GT = REPO / "experiments/reasoning/data/company_reports_gt.json"
DATA = REPO / "experiments/reasoning/data/train_50/sustainability.json"

# run -> (label, group)
RUN_INFO = {
    "head_1000_jev_combined": ("first 1k + Jev filter & classify", "decomp"),
    "head_1000_jev": ("first 1k + Jev classify", "decomp"),
    "head_1000": ("first 1k tokens", "decomp"),
    "head_2000": ("first 2k tokens", "decomp"),
    "head_4000": ("first 4k tokens", "decomp"),
    "head2000_bm25": ("first 2k + BM25 to 4k", "decomp"),
    "bm25_c500": ("BM25 500-token chunks, 4k", "decomp"),
    "plan_13": ("plan 13", "search"),
    "plan_15": ("plan 15", "search"),
    "plan_17": ("plan 17", "search"),
    "plan_20": ("plan 20", "search"),
    "root": ("full text", "root"),
}
# Reference palette, light mode, categorical slots 1-3 (validated all-pairs for scatter).
GROUPS = {
    "decomp": ("Data decomposition (this study)", "#2a78d6"),
    "search": ("MOAR search plans", "#eb6834"),
    "root": ("Original pipeline", "#1baf7a"),
}
YLIM = {"activity": (0.63, 0.73), "combined": (0.76, 0.86)}
SURFACE, TEXT, TEXT_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def score():
    rows = []
    for run, (label, group) in RUN_INFO.items():
        out = RUNS / run / "output.json"
        if not out.exists():
            continue
        m = evaluate_results(run, str(out), str(GT), str(DATA))
        cost = json.load(open(RUNS / run / "run_meta.json"))["cost"]
        rows.append({"run": run, "label": label, "group": group, "cost": cost,
                     "activity": m["economic_activity_accuracy"],
                     "names": m["company_name_accuracy"],
                     "combined": m["combined_score"]})
    return rows


def frontier(rows, key):
    """Cheapest-first; a run is on the frontier if it beats every cheaper run."""
    best, out = -1.0, []
    for r in sorted(rows, key=lambda r: (r["cost"], -r[key])):
        if r[key] > best:
            out.append(r)
            best = r[key]
    return out


def ties(rows, front, key, cost_tol=0.02):
    """Runs off the frontier only because they cost < 2% more for the same score."""
    return {r["run"] for r in rows for f in front
            if r["run"] != f["run"] and r[key] == f[key] and r["cost"] <= f["cost"] * (1 + cost_tol)}


def panel(ax, rows, key, title):
    front = frontier(rows, key)
    front_runs = {r["run"] for r in front}
    # Frontier as a step line: accuracy holds until the next frontier point.
    xs, ys = [], []
    for i, r in enumerate(front):
        xs.append(r["cost"]); ys.append(r[key])
        nxt = front[i + 1]["cost"] if i + 1 < len(front) else 3.6
        xs.append(nxt); ys.append(r[key])
    ax.plot(xs, ys, color=TEXT_2, lw=1.5, ls=(0, (4, 3)), zorder=1)

    tied = ties(rows, front, key)
    for r in rows:
        on = r["run"] in front_runs or r["run"] in tied
        ax.scatter(r["cost"], r[key], s=90 if on else 60, color=GROUPS[r["group"]][1],
                   edgecolor=SURFACE, linewidth=2, zorder=3, alpha=1 if on else 0.55)

    # Number each point (numbers follow cost order and are explained in the key below the
    # chart). Points at the same score and nearly the same cost share one label.
    placed = []
    for r in sorted(rows, key=lambda r: r["cost"]):
        same = next((p for p in placed if p["score"] == r[key] and r["cost"] <= p["cost"] * 1.03), None)
        if same:
            same["nums"].append(r["num"])
        else:
            placed.append({"cost": r["cost"], "score": r[key], "nums": [r["num"]]})
    above = []
    for p in placed:
        # Put the number below the point if a label above would crowd a nearby one.
        crowded = any(q["cost"] / p["cost"] < 1.15 and p["cost"] / q["cost"] < 1.15
                      and abs(q["score"] - p["score"]) < 0.012 for q in above)
        if not crowded:
            above.append(p)
        ax.annotate(",".join(str(n) for n in p["nums"]), (p["cost"], p["score"]), textcoords="offset points",
                    xytext=(9, 0) if crowded else (0, 9), ha="left" if crowded else "center",
                    va="center" if crowded else "bottom", fontsize=9, color=TEXT, fontweight="bold")

    ax.set_xscale("log")
    ax.set_xlim(0.018, 3.6)
    ax.xaxis.set_major_locator(FixedLocator([0.02, 0.05, 0.1, 0.2, 0.5, 1, 2]))
    ax.xaxis.set_minor_locator(NullLocator())
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}"))
    ax.set_ylim(*YLIM[key])
    ax.set_title(title, loc="left", fontsize=11, color=TEXT, pad=10)
    ax.set_xlabel("Cost for 50 reports (USD, log scale)", fontsize=9, color=TEXT_2)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8.5, length=0)
    return front


def main():
    rows = score()
    # plan_20 broke under the model swap (0.14). plan_13 groups by the input's company_name
    # field, which is the GT label, and copies it into the output (label leakage).
    usable = [r for r in rows if r["run"] not in ("plan_20", "plan_13")]
    json.dump(rows, open(RUNS / "frontier_scores.json", "w"), indent=1)

    for i, r in enumerate(sorted(usable, key=lambda r: r["cost"]), 1):
        r["num"] = i
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.6), facecolor=SURFACE)
    fronts = {}
    for ax, (key, title) in zip(axes, [
        ("activity", "Activity accuracy (MOAR's search reward)"),
        ("combined", "Combined score (activity + names)"),
    ]):
        ax.set_facecolor(SURFACE)
        fronts[key] = panel(ax, usable, key, title)
    for ax in axes:
        ax.set_ylabel("Score (corrected evaluator)", fontsize=9, color=TEXT_2)
    on_any = {r["run"] for front in fronts.values() for r in front}
    on_any |= {run for key, front in fronts.items() for run in ties(usable, front, key)}
    key_lines = [f"{r['num']}  {r['label']}  ${r['cost']:.3f}" + ("  *" if r["run"] in on_any else "")
                 for r in sorted(usable, key=lambda r: r["num"])]
    half = (len(key_lines) + 1) // 2
    for col, lines in enumerate([key_lines[:half], key_lines[half:]]):
        fig.text(0.07 + 0.45 * col, 0.175, "\n".join(lines), fontsize=8.5, color=TEXT, va="top",
                 linespacing=1.5)

    handles = [plt.Line2D([], [], marker="o", ls="", markersize=8, color=c, markeredgecolor=SURFACE,
                          label=name) for name, c in GROUPS.values()]
    handles.append(plt.Line2D([], [], color=TEXT_2, lw=1.5, ls=(0, (4, 3)), label="Pareto frontier (* in key = on a frontier or tied with one)"))
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=9,
               bbox_to_anchor=(0.06, 0.99), labelcolor=TEXT)
    fig.text(0.07, 0.012, "gemini-3.5-flash-lite (runs 1-2 use jev-1.13.0 for the steps named), train_50 (50 reports), "
             "one run each; 1 report = 0.02.\nNot shown: plan 13 (reads the GT company name from the input), "
             "plan 20 (0.14 after its gpt-5-nano filters were swapped to flash-lite).",
             fontsize=8, color=TEXT_2, linespacing=1.5)
    fig.tight_layout(rect=(0, 0.22, 1, 0.92))
    fig.savefig(RUNS / "frontier.png", dpi=160, facecolor=SURFACE)

    print(f"{'run':16s} {'cost':>7s} {'activity':>8s} {'names':>6s} {'combined':>8s}")
    for r in sorted(rows, key=lambda r: r["cost"]):
        print(f"{r['run']:16s} ${r['cost']:6.3f} {r['activity']:8.2f} {r['names']:6.2f} {r['combined']:8.2f}")
    for key, front in fronts.items():
        print(f"frontier ({key}):", " -> ".join(r["run"] for r in front),
              "| ties:", sorted(ties(usable, front, key)))
    print("wrote", RUNS / "frontier.png")


if __name__ == "__main__":
    main()
