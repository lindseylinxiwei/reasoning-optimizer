"""Model sweep tier G1: the 6 frontier plans on gemini-3.1-flash-lite, 3.5-flash-lite (Levels 1/2) and 3.8-flash
-> runs/level3/report/model_sweep_g1.png. Color = plan, marker = model, one frontier over all points."""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_level1 import AXIS, GRID, INK, INK2, MUTED, SLOTS, SURFACE, frontier

HERE = Path(__file__).resolve().parent
L1, L2, L3 = (HERE / "runs" / d for d in ("level1", "level2", "level3"))
PLANS = {"Base": (L1, "baseline"), "C1": (L2, "C1_gate_jev_threshold_0.2"), "I3a": (L1, "isolating_subtasks_1"),
         "A1": (L2, "A1_gate_jev_isolating"), "J5": (L1, "jev_5"), "H5": (L1, "head_tail_5")}
PLAN_COLOR = {"Base": INK, "C1": SLOTS[4], "I3a": SLOTS[6], "A1": SLOTS[0], "J5": SLOTS[3], "H5": SLOTS[2]}
MODELS = [("g31fl", "3.1-flash-lite", "3.1fl", "v"), ("g35fl", "3.5-flash-lite", "3.5fl", "o"),
          ("g38f", "3.8-flash", "3.8f", "^")]


def points(tags=("g31fl", "g38f")):
    pts = []
    for code, (folder, name) in PLANS.items():
        r = next(x for x in json.load(open(folder / "scores.json")) if x["plan"] == name)
        pts.append(dict(code=code, tag="g35fl", cost=r["cost"], f1=r["f1"], precision=r["precision"],
                        recall=r["recall"], f1_macro=r["f1_macro"], neg=r["negative_task_accuracy"]))
    for r in json.load(open(L3 / "scores.json")):
        code, tag = r["cell"].split("__")
        if tag in tags:
            pts.append(dict(code=code, tag=tag, cost=r["cost"], f1=r["f1"], precision=r["precision"],
                            recall=r["recall"], f1_macro=r["f1_macro"], neg=r["negative_task_accuracy"]))
    return pts


def main():
    pts = points()
    short = {t: s for t, _, s, _ in MODELS}
    offsets = json.load(open(L3 / "report" / "label_offsets_g1.json")) \
        if (L3 / "report" / "label_offsets_g1.json").exists() else {}
    fig, ax = plt.subplots(figsize=(10, 6.4), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    front = frontier([dict(p, plan=f"{p['code']}·{short[p['tag']]}") for p in pts])
    right = max(p["cost"] for p in pts) * 1.5
    ax.step([r["cost"] for r in front] + [right], [r["f1"] for r in front] + [front[-1]["f1"]], where="post",
            color=INK2, lw=1.2, zorder=1, label="Pareto frontier")
    for code, color in PLAN_COLOR.items():  # plan legend entries (color)
        ax.scatter([], [], c=color, marker="s", s=40, label=f"Plan {code}")
    for tag, label, _, marker in MODELS:  # model legend entries (marker)
        ax.scatter([], [], facecolors="none", edgecolors=INK2, marker=marker, s=46, label=label)
    for p in pts:
        marker = next(m for t, _, _, m in MODELS if t == p["tag"])
        ax.scatter(p["cost"], p["f1"], c=PLAN_COLOR[p["code"]], marker=marker, s=58, edgecolors=SURFACE,
                   linewidths=1.2, zorder=3)
        lab = f"{p['code']}·{short[p['tag']]}"
        dx, dy = offsets.get(lab, (6, -10))
        ax.annotate(lab, (p["cost"], p["f1"]), xytext=(dx, dy), textcoords="offset points", fontsize=7.5,
                    color=INK, zorder=4, ha="left" if dx >= 0 else "right")
    ax.set_xscale("log")
    ticks = [t for t in [0.1, 0.2, 0.5, 1, 2, 5] if t <= right]
    ax.set_xticks(ticks); ax.set_xticklabels([f"${t:.2f}" if t < 1 else f"${t:g}" for t in ticks])
    ax.set_xlim(min(p["cost"] for p in pts) * 0.8, right)
    lo = min(p["f1"] for p in pts)
    ax.set_ylim(max(0, lo - 0.05), max(p["f1"] for p in pts) + 0.03)
    ax.set_xlabel("Cost of one run over the 40 train contracts (log scale)", color=INK2, fontsize=10)
    ax.set_ylabel("CUAD F1 (official matching, pooled)", color=INK2, fontsize=10)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    for side in ("left", "bottom"): ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title("CUAD Level 3, tier G1: frontier plans on three Gemini models", loc="left", color=INK, fontsize=12,
                 pad=10)
    ax.legend(loc="lower right", fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=GRID, labelcolor=INK2, ncol=2)
    fig.tight_layout()
    (L3 / "report").mkdir(parents=True, exist_ok=True)
    out = L3 / "report" / "model_sweep_g1.png"
    fig.savefig(out, facecolor=SURFACE)
    print(out, "frontier:", [r["plan"] for r in front])


if __name__ == "__main__":
    main()
