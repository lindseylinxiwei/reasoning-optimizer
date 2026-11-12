"""Readable search-tree log: one line per plan, children indented under their parent.

    #30 take_head_tail(filter_sustainability_reports) head_words=1000  acc  $0.20  0.771 (+0.014)  n=3  ★

Each line: node id, the rewrite that made it (target operations, grid value), goal, cost of the
plan on the train set, score and change vs the parent, visits, ★ if on the Pareto frontier.
Children that were skipped or failed are summarised in one line under their parent.

Used by the search after every iteration (mcts_tree_log.txt), and from the command line on a run
directory: python -m docetl.mcts.tree_log <run_dir>
"""
from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

GUARD_PREFIXES = ("estimated $",)


def _frontier(nodes: Dict) -> set:
    pts = [(n["id"], n["cost"], n["acc"]) for n in nodes.values()
           if n.get("cost") is not None and n["cost"] >= 0 and n.get("acc") is not None]
    return {i for i, c, a in pts
            if not any(c2 <= c and a2 >= a and (c2 < c or a2 > a) for _, c2, a2 in pts)}


def _label(n: Dict) -> str:
    action = n.get("action") or "root"
    if action.startswith("change to "):
        return "model → " + action[len("change to "):].removeprefix("gemini-").removeprefix("gemini/gemini-")
    targets = ",".join(n.get("targets") or [])
    text = f"{action}({targets})" if targets else action
    grid = n.get("grid")
    if grid:
        text += f" {grid.get('param')}={grid.get('value')}"
    return text


def _skip_reason(reason: Optional[str]) -> str:
    reason = reason or "unknown"
    if reason.startswith(GUARD_PREFIXES):
        return "cost guard"
    if "shortening parent" in reason:
        return "shortening parent not run"
    return reason[:40]


def render_tree(nodes: Iterable[Dict], events: Iterable[Dict] = (), header: str = "",
                new_ids: Optional[set] = None) -> str:
    """nodes: dicts with id, parent, action, targets, goal, grid, cost, acc, visits.
    events: rows of nodes.jsonl (to summarise skipped and failed children per parent)."""
    by_id = {n["id"]: n for n in nodes}
    kids: Dict = defaultdict(list)
    for n in by_id.values():
        if n.get("parent") is not None and n["parent"] in by_id:
            kids[n["parent"]].append(n["id"])
    lost: Dict = defaultdict(Counter)
    for e in events:
        if e.get("id") in by_id or e.get("status") == "ok":
            continue
        status = e.get("status")
        why = ("failed" if status == "failed" else "pruned: dominated by a sibling value" if status == "pruned"
               else "skipped: " + _skip_reason(e.get("reason")))
        lost[e.get("parent")][why] += 1
    front = _frontier(by_id)
    new_ids = new_ids or set()
    labels = {i: _label(n) for i, n in by_id.items()}
    rows: List = []         # (tree text, node id) or (summary text, None); columns added at the end

    def line(i: int, prefix: str, connector: str) -> None:
        rows.append((f"{prefix}{connector}#{i} {labels[i]}", i))

    def columns(i: int) -> str:
        n = by_id[i]
        cols = []
        cols.append(f"{n.get('goal') or '':4s}")
        cost, acc = n.get("cost"), n.get("acc")
        cols.append(f"{'$' + format(cost, '.2f'):>7}" if cost is not None and cost >= 0 else " " * 7)
        if acc is not None:
            p = by_id.get(n.get("parent"))
            d = f" ({acc - p['acc']:+.3f})" if p and p.get("acc") is not None else " " * 9
            cols.append(f"{acc:.3f}{d}")
        else:
            cols.append(" " * 14)
        cols.append(f"n={n.get('visits') or 0}")
        mark = ("★" if i in front else " ") + ("+" if i in new_ids else " ")
        return f"{'  '.join(cols)}  {mark}"

    def walk(i: int, prefix: str, connector: str, child_prefix: str) -> None:
        line(i, prefix, connector)
        children = sorted(kids.get(i, []))
        summary = ", ".join(f"{c} {w}" for w, c in sorted(lost.get(i, {}).items()))
        for k, c in enumerate(children):
            last = k == len(children) - 1 and not summary
            walk(c, child_prefix, "└─ " if last else "├─ ", child_prefix + ("   " if last else "│  "))
        if summary:
            rows.append((f"{child_prefix}└─ ({summary})", None))

    roots = [i for i, n in by_id.items() if n.get("parent") not in by_id]
    for r in sorted(roots):
        walk(r, "", "", "")
    width = max([len(t) for t, i in rows if i is not None] + [20]) + 2
    lines = [header] if header else []
    lines.append("★ Pareto frontier   + new this iteration   acc/cost = goal   n = visits")
    for text, i in rows:
        lines.append(f"{text:<{width}}{columns(i)}".rstrip() if i is not None else text)
    return "\n".join(lines) + "\n"


def nodes_from_checkpoint(state: Dict, events: List[Dict]) -> List[Dict]:
    targets = {e["id"]: e.get("target_ops") for e in events if e.get("status") == "ok"}
    out = []
    for r in state["nodes"]:
        memo = r.get("memo") or []
        out.append({
            "id": r["id"], "parent": r.get("parent"), "action": r.get("latest_action"),
            "targets": targets.get(r["id"]) or ([memo[-1][1]] if memo else []),
            "goal": r.get("goal"), "grid": r.get("grid"),
            "cost": r.get("cost") if (r.get("cost") is not None and r["cost"] >= 0) else None,
            "acc": r.get("accuracy"), "visits": r.get("visits"),
        })
    return out


def render_run_dir(run_dir: str) -> str:
    d = Path(run_dir)
    state = json.loads((d / "checkpoint.json").read_text())
    events = [json.loads(l) for l in (d / "nodes.jsonl").read_text().splitlines() if l.strip()] \
        if (d / "nodes.jsonl").exists() else []
    spent = ""
    if (d / "costs.jsonl").exists():
        totals: Counter = Counter()
        for l in (d / "costs.jsonl").read_text().splitlines():
            if l.strip():
                r = json.loads(l)
                totals[r["kind"]] += r["usd"]
        spent = (f" · spent ${sum(totals.values()):.2f} (agent ${totals['agent']:.2f}, "
                 f"pipelines ${totals['pipeline']:.2f})")
    header = f"After iteration {state.get('iteration_count')}{spent}"
    return render_tree(nodes_from_checkpoint(state, events), events, header)


if __name__ == "__main__":
    print(render_run_dir(sys.argv[1]), end="")
