"""Pareto frontier over every plan in the shared-data-view sweep (one chart, two metrics).

Technique is shown by marker shape (one neutral color), plans on the frontier in blue, the
full-text pipeline as a reference, and every point carries a short code (H1..H5, K1..K5,
K3-bm25, K3-half, J1..J5, J3-adapt, S1..S5). Writes runs/view_sweep/frontier_all.png.
"""
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "runs" / "view_sweep"
SURFACE, TEXT, TEXT_2, GRID, POINT = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#7d7c76"
FRONT = "#2a78d6"
MARKERS = {"head_tail": ("o", "Head/tail"), "topk": ("s", "Top-k retrieval"),
           "jev": ("^", "Jev retrieval"), "summarization": ("D", "Summarization")}
FULL_TEXT = {"plan": "full_text", "cost": 2.402, "combined": 0.81, "activity": 0.69}


def code(plan):
    special = {"topk_3_fts": "K3-bm25", "topk_3_half_chunks": "K3-half", "jev_3_adaptive": "J3-adapt"}
    if plan in special:
        return special[plan]
    tech, level = plan.rsplit("_", 1)
    return {"head_tail": "H", "topk": "K", "jev": "J", "summarization": "S"}[tech] + level


def frontier(rows, key):
    best, out = -1.0, []
    for r in sorted(rows, key=lambda r: (r["cost"], -r[key])):
        if r[key] > best:
            out.append(r)
            best = r[key]
    return out


def place_labels(ax, fig, points, key, front_y):
    """Place each code at the first candidate offset whose rendered box overlaps no point, no
    earlier label and not the frontier line (all checked in display coordinates)."""
    renderer = fig.canvas.get_renderer()
    to_px = ax.transData.transform
    dots = [to_px((r["cost"], r[key])) for r in points] + [to_px((FULL_TEXT["cost"], FULL_TEXT[key]))]
    line_y = to_px((1, front_y))[1]
    taken = []
    candidates = [(8, 0), (0, -11), (8, -10), (8, 10), (-8, 0), (0, 11), (-8, -10), (-8, 10),
                  (14, -18), (14, 18), (-14, -18), (-14, 18), (0, -20), (0, 20)]
    for r in sorted(points, key=lambda r: (-r[key], r["cost"])):
        for dx, dy in candidates:
            ha = "left" if dx > 0 else "right" if dx < 0 else "center"
            va = "center" if dy == 0 else ("bottom" if dy > 0 else "top")
            t = ax.annotate(code(r["plan"]), (r["cost"], r[key]), textcoords="offset points", xytext=(dx, dy),
                            ha=ha, va=va, fontsize=10.5, color=TEXT)
            box = t.get_window_extent(renderer).expanded(1.08, 1.15)
            hits_dot = any(box.x0 - 4 < x < box.x1 + 4 and box.y0 - 4 < y < box.y1 + 4 for x, y in dots)
            hits_label = any(box.overlaps(b) for b in taken)
            hits_line = box.y0 - 1 < line_y < box.y1 + 1
            if not (hits_dot or hits_label or hits_line):
                taken.append(box)
                break
            t.remove()
        else:  # no free spot: keep the first candidate
            taken.append(ax.annotate(code(r["plan"]), (r["cost"], r[key]), textcoords="offset points",
                                     xytext=(8, 0), ha="left", va="center", fontsize=10.5,
                                     color=TEXT).get_window_extent(renderer))


def main():
    rows = json.load(open(OUT / "scores.json"))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), facecolor=SURFACE)
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
        for tech, (marker, _) in MARKERS.items():
            mine = [r for r in rows if r["technique"] == tech]
            for r in mine:
                size = {"^": 1.5, "D": 0.8}.get(marker, 1.0) * (80 if r["plan"] in on else 56)
                ax.scatter(r["cost"], r[key], marker=marker, s=size,
                           color=FRONT if r["plan"] in on else POINT, edgecolor=SURFACE, linewidth=1.5, zorder=3)
        ax.scatter(FULL_TEXT["cost"], FULL_TEXT[key], marker="*", s=150, color=TEXT, zorder=3)
        ax.annotate("full text", (FULL_TEXT["cost"], FULL_TEXT[key]), textcoords="offset points",
                    xytext=(0, -12), ha="center", va="top", fontsize=11, color=TEXT)
        ax.set_xscale("log")
        ax.set_xlim(0.035, 3.2)
        fig.canvas.draw()
        place_labels(ax, fig, rows, key, front[0][key])
        ax.xaxis.set_major_locator(FixedLocator([0.05, 0.1, 0.2, 0.5, 1, 2]))
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
    handles = [plt.Line2D([], [], ls="", marker=m, markersize=7, color=POINT, label=l) for m, l in MARKERS.values()]
    handles += [plt.Line2D([], [], ls="", marker="o", markersize=8, color=FRONT, label="On the Pareto frontier"),
                plt.Line2D([], [], color=FRONT, lw=1.5, ls=(0, (4, 3)), label="Pareto frontier"),
                plt.Line2D([], [], ls="", marker="*", markersize=10, color=TEXT, label="Full text (no decomposition)")]
    fig.legend(handles=handles, loc="upper left", ncol=4, frameon=False, fontsize=11.5,
               bbox_to_anchor=(0.04, 0.995), labelcolor=TEXT)
    fig.text(0.04, 0.008, "Codes: H = head/tail, K = top-k, J = Jev, S = summarization; number = level (1 = least text).\n"
             "gemini-3.5-flash-lite, train_50, corrected evaluator, one run each (1 report = about 0.023).",
             fontsize=11, color=TEXT_2)
    fig.tight_layout(rect=(0, 0.07, 1, 0.89))
    fig.savefig(OUT / "frontier_all.png", dpi=110, facecolor=SURFACE)
    print("wrote", OUT / "frontier_all.png")


if __name__ == "__main__":
    main()
