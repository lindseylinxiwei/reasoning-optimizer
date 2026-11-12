"""Grid variants of one rewrite: only the Pareto-optimal ones (tie goes to cheaper) stay in the tree."""
from types import SimpleNamespace

from docetl.mcts.mcts import MCTS
from docetl.mcts.param_grid import ParamTable


class N:
    def __init__(self, id, parent, value):
        self.id, self.parent, self.children = id, parent, []
        self.grid = {"directive": "take_head_tail", "param": "head_words", "value": value,
                     "op": "filter", "parent_model": "gemini-3-flash-preview"}
        if parent is not None:
            parent.children.append(self)

    def get_id(self):
        return self.id


def run(results):
    parent = N(3, None, None)
    kids = {v: N(i, parent, v) for i, (v, _, _) in zip((29, 30, 31), results)}
    logged = []
    tree = SimpleNamespace(param_table=ParamTable(), tree_lock=__import__("threading").RLock(),
                           _log_node=lambda c, cost, acc, status, reason=None: logged.append((c.id, status)))
    runs = [(kids[v], c, a) for v, c, a in results]
    kept = MCTS._keep_pareto_siblings(tree, runs)
    return [k[0].id for k in kept], [c.id for c in parent.children], logged, tree.param_table


def test_test6_grid_keeps_only_the_cheap_winner():
    kept, children, logged, table = run([(2000, 0.29, 0.750), (1000, 0.20, 0.771), (4000, 0.49, 0.730)])
    assert kept == [30] and children == [30]
    assert sorted(logged) == [(29, "pruned"), (31, "pruned")]
    assert len(table._results("take_head_tail", "filter", "gemini-3-flash-preview")) == 2   # dropped values recorded


def test_a_cheaper_clearly_worse_value_is_kept_as_a_tradeoff():
    kept, children, _, _ = run([(1000, 0.10, 0.60), (2000, 0.30, 0.75), (4000, 0.60, 0.78)])
    assert sorted(kept) == [29, 30]          # 4000 ties 2000 (within 0.05) and costs more
    assert sorted(children) == [29, 30]
