"""Cost vs. F1 for CUAD Level 2 (runs/level2/scores.json) over Level 1 -> runs/level2/report/frontier_level2.png."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_level1 import AXIS, GRID, INK, INK2, MUTED, SLOTS, SURFACE, frontier

HERE = Path(__file__).resolve().parent
L1, L2 = HERE / "runs" / "level1", HERE / "runs" / "level2"
FAINT = "#c3c2b7"

CODES = {
    "A1_gate_jev_isolating": "A1", "A1b_gate_jev_t02_isolating": "A1-t0.2", "A2_gate_llm_isolating": "A2", "A3_chunk16k_gate_per_chunk": "A3",
    "A5_chunk16k_isolating": "A5",
    "R1_isolating_bm25_per_group_35": "R1", "R2_isolating_bm25_per_group_70": "R2",
    "R3_isolating_emb_per_group_35": "R3", "R4_isolating_emb_per_group_70": "R4",
    "R5_isolating_jev_per_group_35": "R5", "R6_isolating_jev_per_group_70": "R6",
    "R7_view_jev5_isolating": "R7", "R8_view_bm25_5_isolating": "R8", "R9_view_emb5_isolating": "R9", "R10_isolating_jev_broad_cs_c3k_long": "R10",
    "B1_view_jev5_gate": "B1", "B2_view_bm25_5_gate": "B2", "B3_view_head5_gate": "B3",
    "C1_gate_jev_threshold_0.2": "C1", "C2_gate_jev_threshold_0.5": "C2",
    "D_baseline_rerun": "D-Base", "D_gate_jev_2_rerun": "D-GJ2", "D_gate_llm_1_rerun": "D-GL1",
    "D_isolating_subtasks_1_rerun": "D-I3a", "D_isolating_subtasks_2_rerun": "D-I3b",
}
# (legend label, color, marker, family prefix)
GROUPS = [
    ("A: task + task", SLOTS[0], "o", "A:"),
    ("R: isolating + per-group retrieval", SLOTS[1], "o", "R:isolating+"),
    ("R: global view + isolating", SLOTS[3], "s", "R:global_view"),
    ("B: view + Jev gate", SLOTS[2], "o", "B:"),
    ("C: Jev gate threshold", SLOTS[4], "D", "C:"),
    ("D: rerun of a Level 1 plan", INK2, "^", "D:"),
]


def l1_code(plan):
    from plot_level1 import code
    return code(plan)


def main():
    r1 = json.load(open(L1 / "scores.json"))
    r2 = json.load(open(L2 / "scores.json"))
    fig, ax = plt.subplots(figsize=(10, 6.4), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    right = 2.6
    f1 = frontier(r1)
    ax.step([r["cost"] for r in f1] + [right], [r["f1"] for r in f1] + [f1[-1]["f1"]], where="post",
            color=FAINT, lw=1.2, zorder=1, label="Level 1 frontier")
    ax.scatter([r["cost"] for r in r1], [r["f1"] for r in r1], s=22, c=FAINT, edgecolors="none", zorder=2,
               label="Level 1 plan")
    both = [dict(r, level=1) for r in r1] + [dict(r, level=2) for r in r2]
    fb = frontier(both)
    ax.step([r["cost"] for r in fb] + [right], [r["f1"] for r in fb] + [fb[-1]["f1"]], where="post",
            color=INK, lw=1.3, zorder=3, label="Level 1 + 2 frontier")
    for label, color, marker, prefix in GROUPS:
        g = [r for r in r2 if r["family"].startswith(prefix)]
        if g:
            ax.scatter([r["cost"] for r in g], [r["f1"] for r in g], s=50, c=color, marker=marker,
                       edgecolors=SURFACE, linewidths=1.4, zorder=4, label=label)
    offsets = json.load(open(L2 / "report" / "label_offsets.json")) if (L2 / "report" / "label_offsets.json").exists() else {}
    for r in r2:
        c = CODES.get(r["plan"], r["plan"])
        dx, dy = offsets.get(c, (5, 5))
        ax.annotate(c, (r["cost"], r["f1"]), xytext=(dx, dy), textcoords="offset points", fontsize=7.5,
                    color=INK, zorder=5, ha="left" if dx >= 0 else "right")
    for r in fb:
        if r["level"] == 1:
            c = l1_code(r["plan"])
            dx, dy = offsets.get(f"L1:{c}", (5, -10))
            ax.annotate(c, (r["cost"], r["f1"]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=7.5, color=MUTED, zorder=5, ha="left" if dx >= 0 else "right")
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2]); ax.set_xticklabels(["$0.05", "$0.10", "$0.20", "$0.50", "$1", "$2"])
    ax.set_xlim(0.05, right); ax.set_ylim(0.1, 0.66)
    ax.set_xlabel("Cost of one run over the 40 train contracts (log scale)", color=INK2, fontsize=10)
    ax.set_ylabel("CUAD F1 (official matching, pooled)", color=INK2, fontsize=10)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    for side in ("left", "bottom"): ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(f"CUAD Level 2: {len(r2)} combination plans over the 45 Level 1 plans", loc="left", color=INK,
                 fontsize=12, pad=10)
    ax.legend(loc="lower right", fontsize=8.5, frameon=True, facecolor=SURFACE, edgecolor=GRID, labelcolor=INK2)
    fig.tight_layout()
    (L2 / "report").mkdir(parents=True, exist_ok=True)
    out = L2 / "report" / "frontier_level2.png"
    fig.savefig(out, facecolor=SURFACE)
    print(out, "combined frontier:", [CODES.get(r["plan"]) or l1_code(r["plan"]) for r in fb])


if __name__ == "__main__" and "--on-level1" not in __import__("sys").argv:
    main()


# ---- Level 1 figure (full colors) with the Level 2 plans added -> frontier_level1_with_level2.png ----------
L2_MARKERS = [  # (legend label, marker, family prefix); Level 2 plans are drawn in ink so Level 1's colors keep their meaning
    ("Level 2: A, task + task", "P", "A:"),
    ("Level 2: R, isolating + per-group retrieval", "v", "R:isolating+"),
    ("Level 2: R, global view + isolating", "<", "R:global_view"),
    ("Level 2: B, view + Jev gate", ">", "B:"),
    ("Level 2: C, Jev gate threshold", "D", "C:"),
    ("Level 2: D, rerun of a Level 1 plan", "^", "D:"),
]


def level1_with_level2():
    from plot_level1 import GROUPS as L1_GROUPS, code
    r1 = json.load(open(L1 / "scores.json"))
    r2 = json.load(open(L2 / "scores.json"))
    right = 2.6
    fig, ax = plt.subplots(figsize=(10.5, 6.8), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    f1 = frontier(r1)
    ax.step([r["cost"] for r in f1] + [right], [r["f1"] for r in f1] + [f1[-1]["f1"]], where="post",
            color=MUTED, lw=1.1, ls="--", zorder=1, label="Level 1 frontier")
    both = [dict(r, level=1) for r in r1] + [dict(r, level=2) for r in r2]
    fb = frontier(both)
    ax.step([r["cost"] for r in fb] + [right], [r["f1"] for r in fb] + [fb[-1]["f1"]], where="post",
            color=INK, lw=1.4, zorder=1, label="Level 1 + 2 frontier")
    for label, color, marker, pred in L1_GROUPS:
        g = [r for r in r1 if pred(r["plan"])]
        ax.scatter([r["cost"] for r in g], [r["f1"] for r in g], s=90 if marker == "*" else 34, c=color,
                   marker=marker, edgecolors=SURFACE, linewidths=1.2, zorder=3, label=f"Level 1: {label}")
    for label, marker, prefix in L2_MARKERS:
        g = [r for r in r2 if r["family"].startswith(prefix)]
        ax.scatter([r["cost"] for r in g], [r["f1"] for r in g], s=58, c=INK, marker=marker,
                   edgecolors=SURFACE, linewidths=1.2, zorder=4, label=label)
    offsets = json.load(open(L2 / "report" / "label_offsets.json"))
    for r in r2:
        c = CODES[r["plan"]]
        dx, dy = offsets.get(c, (5, 5))
        ax.annotate(c, (r["cost"], r["f1"]), xytext=(dx, dy), textcoords="offset points", fontsize=7.5,
                    color=INK, zorder=5, ha="left" if dx >= 0 else "right")
    for r in fb:
        if r["level"] == 1:
            c = code(r["plan"])
            dx, dy = offsets.get(f"L1:{c}", (5, -10))
            ax.annotate(c, (r["cost"], r["f1"]), xytext=(dx, dy), textcoords="offset points", fontsize=7.5,
                        color=INK2, zorder=5, ha="left" if dx >= 0 else "right")
    ax.set_xscale("log")
    ax.set_xticks([0.05, 0.1, 0.2, 0.5, 1, 2]); ax.set_xticklabels(["$0.05", "$0.10", "$0.20", "$0.50", "$1", "$2"])
    ax.set_xlim(0.05, right); ax.set_ylim(0.1, 0.66)
    ax.set_xlabel("Cost of one run over the 40 train contracts (log scale)", color=INK2, fontsize=10)
    ax.set_ylabel("CUAD F1 (official matching, pooled)", color=INK2, fontsize=10)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    for side in ("left", "bottom"): ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.set_title(f"CUAD: cost against F1 for the 45 Level 1 and {len(r2)} Level 2 plans", loc="left", color=INK,
                 fontsize=12, pad=10)
    ax.legend(loc="lower right", fontsize=7.4, frameon=True, facecolor=SURFACE, edgecolor=GRID, labelcolor=INK2,
              ncol=2, columnspacing=1.0)
    fig.tight_layout()
    out = L2 / "report" / "frontier_level1_with_level2.png"
    fig.savefig(out, facecolor=SURFACE)
    print(out)


if __name__ == "__main__" and "--on-level1" in __import__("sys").argv:
    level1_with_level2()
