"""An iteration that evaluates no plan leaves no visit behind; a node the budget can't afford is blocked."""
import threading

from docetl.mcts.mcts import MCTS


class N:
    def __init__(self, id, parent=None, cost=1.0, visits=0):
        self.id, self.parent, self.cost, self.visits, self.children = id, parent, cost, visits, []
        if parent is not None:
            parent.children.append(self)

    def get_id(self):
        return self.id

    def update_visit(self):
        self.visits += 1


class Tree:
    _undo_visits = MCTS._undo_visits
    _block_if_unaffordable = MCTS._block_if_unaffordable
    increment_visits_up_tree = MCTS.increment_visits_up_tree

    def __init__(self, root, spent, budget=500.0, fraction=0.10):
        self.root, self.total_search_cost, self.max_budget_usd = root, spent, budget
        self.cost_guard_fraction, self.tree_lock, self.action_space_exhausted = fraction, threading.RLock(), False

    def _all_nodes(self):
        out, stack = [], [self.root]
        while stack:
            n = stack.pop()
            out.append(n)
            stack.extend(n.children)
        return out


def test_visits_are_restored():
    root = N(0, visits=6); a = N(1, root, visits=1); b = N(2, root, visits=3)
    t = Tree(root, spent=0.0)
    t.increment_visits_up_tree(a); t.increment_visits_up_tree(a)       # first-layer dual expansion
    t._undo_visits(a, 2)
    assert (root.visits, a.visits, b.visits) == (6, 1, 3)


def test_block_only_when_unaffordable_without_reservations():
    root = N(0); argon = N(6, root, cost=12.31); cheap = N(1, root, cost=1.6)
    kid = N(99, None); kid.guard_estimate = 12.31
    t = Tree(root, spent=400.0)                    # 10% of $100 left = $10 < $12.31
    t._block_if_unaffordable(argon, [kid])
    assert getattr(argon, "no_actions", False) and not t.action_space_exhausted   # cheap leaf still open
    t2 = Tree(root, spent=300.0)                   # 10% of $200 = $20: skipped only due to reservations
    argon2 = N(7, root, cost=12.31)
    t2._block_if_unaffordable(argon2, [kid])
    assert not getattr(argon2, "no_actions", False)
