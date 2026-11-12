"""Search state that survives a crash: checkpoint, per-node log, cost ledger, and the hook
that lets the grid fan-out re-apply an agent's rewrite with another parameter value.

Files in the search output directory:
  checkpoint.json    full tree state, rewritten atomically after every iteration
  nodes.jsonl        one row per evaluated (or skipped) child, append-only
  costs.jsonl        one row per dollar spent (agent or pipeline), append-only; on resume the
                     budget is the sum of this file, so work lost in a crash is still counted
  param_table.jsonl  the grid results table (param_grid.ParamTable)
"""
from __future__ import annotations

import json
import math
import os
import threading
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

CHECKPOINT = "checkpoint.json"
NODES_LOG = "nodes.jsonl"
COSTS_LOG = "costs.jsonl"
PARAM_TABLE = "param_table.jsonl"

# ---- capture of directive.apply calls ----------------------------------------------------

_capture = threading.local()
_installed: set = set()
_install_lock = threading.Lock()


def install_apply_capture(directive) -> None:
    """Wrap the directive class's apply so calls made inside capture_applies() are recorded."""
    cls = type(directive)
    with _install_lock:
        if cls in _installed:
            return
        original = cls.apply

        def apply(self, *args, **kwargs):
            calls = getattr(_capture, "calls", None)
            if calls is not None:
                calls.append((self.name, args, kwargs))
            return original(self, *args, **kwargs)

        cls.apply = apply
        _installed.add(cls)


class capture_applies:
    """Context manager: records every captured apply call made by this thread."""

    def __enter__(self) -> List[tuple]:
        self._prev = getattr(_capture, "calls", None)
        _capture.calls = []
        return _capture.calls

    def __exit__(self, *exc):
        _capture.calls = self._prev
        return False


def find_rewrite(call) -> Optional[BaseModel]:
    _, args, kwargs = call
    for v in list(args) + list(kwargs.values()):
        if isinstance(v, BaseModel):
            return v
    return None


def reapply(directive, call, new_rewrite: BaseModel):
    """Call the directive's apply again with the same arguments and a different rewrite."""
    _, args, kwargs = call
    old = find_rewrite(call)
    args = [new_rewrite if a is old else a for a in args]
    kwargs = {k: (new_rewrite if v is old else v) for k, v in kwargs.items()}
    return type(directive).apply(directive, *args, **kwargs)


# ---- append-only logs ----------------------------------------------------------------------

_log_lock = threading.Lock()


def append_jsonl(path: str, row: dict) -> None:
    with _log_lock:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a") as f:
            f.write(json.dumps(row, default=str) + "\n")
            f.flush()
            os.fsync(f.fileno())


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                pass        # a line cut off by a crash
    return rows


def ledger_totals(path: str) -> Dict[str, float]:
    totals = {"agent": 0.0, "pipeline": 0.0}
    for row in read_jsonl(path):
        totals[row.get("kind", "pipeline")] = totals.get(row.get("kind", "pipeline"), 0.0) + float(row["usd"])
    return totals


# ---- checkpoint ------------------------------------------------------------------------------

def _num(x):
    if x is None:
        return None
    if isinstance(x, float) and (math.isinf(x) or math.isnan(x)):
        return str(x)
    return x


def _unnum(x):
    if isinstance(x, str) and x in ("inf", "-inf", "nan"):
        return float(x)
    return x


def node_record(node, accuracy) -> dict:
    return {
        "id": node.id,
        "parent": node.parent.id if node.parent is not None else None,
        "yaml": str(node.yaml_file_path),
        "cost": _num(node.cost),
        "value": _num(node.value),
        "visits": node.visits,
        "accuracy": _num(accuracy),
        "latest_action": getattr(node.latest_action, "name", None),
        "goal": node.optimization_goal,
        "memo": [list(m) for m in node.memo],
        "used_actions": {op: sorted(a.name for a in acts) for op, acts in node.used_actions.items()},
        "message_history": node.message_history,
        "sample_result": node.sample_result,
        "grid": getattr(node, "grid", None),
        "no_actions": bool(getattr(node, "no_actions", False)),
        "exhausted_goals": sorted(getattr(node, "exhausted_goals", set()) or []),
    }


def write_checkpoint(path: str, state: dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(state, f, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def read_checkpoint(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        state = json.load(f)
    for n in state["nodes"]:
        for k in ("cost", "value", "accuracy"):
            n[k] = _unnum(n[k])
    return state
