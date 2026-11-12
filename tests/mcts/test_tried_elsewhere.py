"""Rewrites chosen in other plans are shown when expanding a plan with the same operations."""
import threading
from types import SimpleNamespace

from docetl.mcts.mcts import MCTS


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)

    def get_id(self):
        return self.id


def make_tree():
    t = Obj(attempts=[], tree_lock=threading.RLock(), _attempt_local=threading.local(), iteration_count=0)
    for name in ("_register_attempt", "_attempt_result", "_tried_elsewhere_text"):
        setattr(t, name, getattr(MCTS, name).__get__(t))
    t._rewrite_targets = MCTS._rewrite_targets
    t._grid_label = MCTS._grid_label
    return t


def test_results_in_progress_and_skips_are_listed_for_other_plans():
    t = make_tree()
    ops = {"filter_reports": {}, "determine_activity": {}, "summary": {}}
    p1 = Obj(id=1, parent=None, cost=1.6, memo=[], op_dict=ops)
    p2 = Obj(id=2, parent=None, cost=2.0, memo=[], op_dict=ops)
    p3 = Obj(id=3, parent=None, cost=3.4, memo=[], op_dict=ops)
    t.pareto_frontier = SimpleNamespace(plans_accuracy={p1: 0.722, p2: 0.73, p3: 0.757})
    chain = Obj(name="reduce_chaining")
    t._register_attempt(p1, "reduce_chaining", ["summary"], "acc")
    t._register_attempt(p2, "reduce_chaining", ["summary"], "acc")
    t._register_attempt(p3, "take_head_tail", ["filter_reports"], "cost")
    c7 = Obj(id=7, parent=p1, latest_action=chain, memo=[("reduce_chaining", "summary")], grid=None)
    t._attempt_result(c7, 1.63, 0.75, "ok", None)
    c8 = Obj(id=8, parent=p2, latest_action=chain, memo=[("reduce_chaining", "summary")], grid=None)
    t._attempt_result(c8, None, None, "skipped", "estimated $2.03 is more than 10% of the remaining $17.84")

    text = t._tried_elsewhere_text(p3)
    assert "reduce_chaining on summary (plan #1, acc goal): score +0.028, cost +$0.03" in text
    assert "reduce_chaining on summary (plan #2, acc goal): not run (cost guard)" in text
    assert "take_head_tail" not in text                      # the plan's own rewrite is not repeated
    assert "in progress" in t._tried_elsewhere_text(p1)      # p3's take_head_tail has no result yet
    other = Obj(id=9, parent=None, cost=1.0, memo=[], op_dict={"unrelated": {}})
    assert t._tried_elsewhere_text(other) == ""


def test_failed_instantiation_is_closed():
    t = make_tree()
    p = Obj(id=1, parent=None, cost=1.0, memo=[], op_dict={"m": {}})
    q = Obj(id=2, parent=None, cost=1.0, memo=[], op_dict={"m": {}})
    t.pareto_frontier = SimpleNamespace(plans_accuracy={})

    def failing_expand(leaf, optimize_goal):
        t._register_attempt(leaf, "chaining", ["m"], optimize_goal)
        raise RuntimeError("All 1 instantiation(s) failed")
    t.expand = failing_expand
    try:
        MCTS._expand_tracked(t, p, "acc")
    except RuntimeError:
        pass
    assert "chaining on m (plan #1, acc goal): produced no plan" in t._tried_elsewhere_text(q)
