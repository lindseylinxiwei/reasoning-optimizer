"""Per-rewrite stats shown to the agent count only children that ran successfully."""
import threading
from types import SimpleNamespace

from docetl.mcts.mcts import MCTS


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_only_successful_children_count():
    a, b = Obj(name="reduce_chaining"), Obj(name="take_head_tail")
    parent = Obj(cost=2.0, parent=None, latest_action=None)
    ok1 = Obj(cost=3.0, parent=parent, latest_action=a)
    ok2 = Obj(cost=1.0, parent=parent, latest_action=a)
    failed = Obj(cost=-1, parent=parent, latest_action=a)
    unscored = Obj(cost=0.5, parent=parent, latest_action=b)      # never got an accuracy
    acc = {parent: 0.70, ok1: 0.75, ok2: 0.71, failed: float("-inf")}
    tree = Obj(available_actions=[a, b], action_outcomes={}, tree_lock=threading.RLock(),
               pareto_frontier=SimpleNamespace(plans_accuracy=acc))
    for child in (ok1, ok2, failed, unscored):
        MCTS._add_outcome(tree, child)
    assert tree.action_counts == {a: 2.0, b: 0.0}
    assert abs(tree.action_cost_changes[a]) < 1e-9                 # +1.0 and -1.0
    assert abs(tree.action_accuracy_changes[a] - 0.06) < 1e-9     # +0.05 and +0.01
