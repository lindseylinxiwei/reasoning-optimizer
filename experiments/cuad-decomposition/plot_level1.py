"""Cost vs. F1 for the CUAD Level 1 sweep (runs/level1/scores.json) -> runs/level1/report/frontier_level1.png.

With --with-moar, also plots the original MOAR search's Pareto-frontier plans that have saved outputs
(runs/level1/report/moar_search_rescored.json: the 8 frontier plans of the MOAR CUAD search log in
~/Downloads/cuad, run on the same 40 contracts, rescored with the new evaluator; cost as MOAR recorded it)
-> runs/level1/report/frontier_level1_with_moar.png. The frontier line stays over Level 1 plans only.
"""
import sys
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
L1 = HERE / "runs" / "level1"
SURFACE, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
# Reference categorical palette (dataviz skill, light mode), slots in fixed order.
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6"]


def code(plan):
    p = plan
    if p == "baseline": return "Base"
    if p.startswith("head_tail_"): return "H" + p[-1]
    if p.startswith("topk_bm25_"): return "B" + p[-1]
    if p == "topk_emb_3_half_chunks": return "E3-half"
    if p.startswith("topk_emb_"): return "E" + p[-1]
    if p == "jev_3_adaptive": return "J3-adapt"
    if p.startswith("jev_"): return "J" + p[-1]
    if p.startswith("gate_jev_"): return "GJ" + p[-1]
    if p.startswith("gate_llm_"): return "GL" + p[-1]
    if p.startswith("isolating_subtasks_g"): return "I" + p.split("_g")[1]
    if p.startswith("isolating_subtasks_"): return "I3" + "ab"[int(p[-1]) - 1]
    if p.startswith("isolating_structured_g"): return "IS" + p.split("_g")[1]
    if p == "isolating_structured_1": return "IS3"
    if p == "chunking_agent_10000": return "C10k-agent"
    if p.startswith("chunking_"): return "C" + str(int(p.split("_")[1]) // 1000) + "k"
    if p.startswith("chunk_header_"): return "CH" + str(int(p.split("_")[-1]) // 1000) + "k"
    return p


# (legend label, color, marker, predicate)
GROUPS = [
    ("Head/tail view", SLOTS[0], "o", lambda p: p.startswith("head_tail")),
    ("Top-k view, BM25", SLOTS[1], "o", lambda p: p.startswith("topk_bm25")),
    ("Top-k view, embedding", SLOTS[2], "o", lambda p: p.startswith("topk_emb")),
    ("Jev retrieval view", SLOTS[3], "o", lambda p: p.startswith("jev_")),
    ("Presence gate, Jev checker", SLOTS[4], "o", lambda p: p.startswith("gate_jev")),
    ("Presence gate, LLM checker", SLOTS[4], "s", lambda p: p.startswith("gate_llm")),
    ("Isolating subtasks, LLM combine", SLOTS[6], "o", lambda p: p.startswith("isolating_subtasks")),
    ("Isolating subtasks, code merge", SLOTS[6], "D", lambda p: p.startswith("isolating_structured")),
    ("Chunking", SLOTS[5], "o", lambda p: p.startswith("chunking")),
    ("Chunking + headers", SLOTS[5], "^", lambda p: p.startswith("chunk_header")),
    ("Baseline (full contract, one map)", INK, "*", lambda p: p == "baseline"),
]


def frontier(rows):
    best, out = -1, []
    for r in sorted(rows, key=lambda r: (r["cost"], -r["f1"])):
        if r["f1"] > best:
            out.append(r); best = r["f1"]
    return out


RERUN_OFFSETS = {14: (-38, 30), 20: (34, 28), 30: (42, 4), 28: (34, -18), 34: (-48, 14), 32: (-48, -10)}  # points


def add_moar_rerun(ax):
    """MOAR's 8 frontier plans rerun on the 40 contracts in the list format (runs/moar_rerun/scores.json), with
    original models except Gemini 2.5 -> 3.5. Only the rerun points are drawn."""
    rerun = {r["node"]: r for r in json.load(open(HERE / "runs" / "moar_rerun" / "scores.json"))}
    pts = [{"cost": r["cost"], "f1": r["f1"], "plan": n} for n, r in rerun.items()]
    mf = frontier(pts)
    ax.step([r["cost"] for r in mf] + [2.6], [r["f1"] for r in mf] + [mf[-1]["f1"]], where="post", color=MUTED,
            lw=1.1, ls="--", zorder=1, label="MOAR rerun frontier")
    ax.scatter([p["cost"] for p in pts], [p["f1"] for p in pts], s=60, c=INK2, marker="X", edgecolors=SURFACE,
               linewidths=1.2, zorder=3, label="MOAR plan rerun (original models, Gemini 2.5 -> 3.5; list format)")
    for p in pts:
        dx, dy = RERUN_OFFSETS.get(p["plan"], (8, -12))
        ax.annotate(f"M{p['plan']}", (p["cost"], p["f1"]), xytext=(dx, dy), textcoords="offset points", fontsize=8.5,
                    color=INK2, ha="left" if dx > 0 else "right", va="center", zorder=4,
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6, shrinkA=1, shrinkB=4))


def main(with_moar=False, rerun=False):
    rows = json.load(open(L1 / "scores.json"))
    front = frontier(rows)
    fig, ax = plt.subplots(figsize=(10, 6.2), dpi=150)
    fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
    xs = [r["cost"] for r in front]; ys = [r["f1"] for r in front]
    ax.step(xs + [6.0 if with_moar else 2.6], ys + [ys[-1]], where="post", color=INK2, lw=1.2, zorder=1, label="Pareto frontier")
    for label, color, marker, pred in GROUPS:
        g = [r for r in rows if pred(r["plan"])]
        ax.scatter([r["cost"] for r in g], [r["f1"] for r in g], s=110 if marker == "*" else 46, c=color,
                   marker=marker, edgecolors=SURFACE, linewidths=1.6, zorder=3, label=label)
    if rerun:
        add_moar_rerun(ax)
    if with_moar:
        # MOAR search log (~/Downloads/cuad): its 8 Pareto-frontier plans, run on the same 40 contracts with their
        # original models and the old comma-joined output, rescored with the new evaluator; cost as MOAR recorded it.
        moar = json.load(open(L1 / "report" / "moar_search_rescored.json"))
        pts = [{"cost": m["cost"], "f1": m["new_f1"], "node": m["node"]} for m in moar]
        mf = frontier([{"cost": m["cost"], "f1": m["f1"], "plan": m["node"]} for m in pts])
        mx = [r["cost"] for r in mf]; my = [r["f1"] for r in mf]
        ax.step(mx + [5.0], my + [my[-1]], where="post", color=MUTED, lw=1.1, ls="--", zorder=1,
                label="MOAR frontier after rescoring")
        ax.scatter([m["cost"] for m in pts], [m["f1"] for m in pts], s=60, c=INK2, marker="X",
                   edgecolors=SURFACE, linewidths=1.2, zorder=3,
                   label="MOAR search frontier plan (original models, old output format)")
        offsets = {34: (-8, 4), 30: (8, -12), 31: (8, -12), 32: (8, -10), 14: (8, -8), 28: (8, -10), 20: (8, 2), 9: (-8, -14)}
        for m in pts:
            dx, dy = offsets.get(m["node"], (8, -12))
            ax.annotate(f"M{m['node']}", (m["cost"], m["f1"]), xytext=(dx, dy), textcoords="offset points",
                        fontsize=8.5, color=INK2, ha="left" if dx > 0 else "right", zorder=4)
    for r in front:  # direct labels on frontier points only
        ax.annotate(code(r["plan"]), (r["cost"], r["f1"]), xytext=(6, -12), textcoords="offset points",
                    fontsize=9, color=INK, zorder=4)
    ax.set_xscale("log")
    ticks = [0.05, 0.1, 0.2, 0.5, 1, 2] + ([4] if with_moar else [])
    ax.set_xticks(ticks); ax.set_xticklabels(["$0.05", "$0.10", "$0.20", "$0.50", "$1", "$2", "$4"][:len(ticks)])
    ax.set_xlim(0.05, 6.0 if with_moar else 2.6); ax.set_ylim(0.1, 0.65)
    ax.set_xlabel("Cost of one run over the 40 train contracts (log scale)", color=INK2, fontsize=10)
    ax.set_ylabel("CUAD F1 (official matching, pooled)", color=INK2, fontsize=10)
    ax.grid(True, color=GRID, lw=0.6); ax.set_axisbelow(True)
    for side in ("top", "right"): ax.spines[side].set_visible(False)
    for side in ("left", "bottom"): ax.spines[side].set_color(AXIS)
    ax.tick_params(colors=MUTED, labelsize=9)
    title = "CUAD Level 1: cost against F1 for 45 plans" + (
        ", with MOAR's frontier plans rerun in the list format" if rerun else
        ", with MOAR's original frontier plans" if with_moar else "")
    ax.set_title(title, loc="left", color=INK, fontsize=12, pad=10)
    leg = ax.legend(loc="lower right", fontsize=8.5, frameon=True, facecolor=SURFACE, edgecolor=GRID, labelcolor=INK2)
    fig.tight_layout()
    out = L1 / "report" / ("frontier_level1_moar_rerun.png" if rerun else
                           "frontier_level1_with_moar.png" if with_moar else "frontier_level1.png")
    fig.savefig(out, facecolor=SURFACE)
    print(out, "frontier:", [code(r["plan"]) for r in front])


if __name__ == "__main__":
    main(with_moar="--with-moar" in sys.argv, rerun="--with-moar-rerun" in sys.argv)
