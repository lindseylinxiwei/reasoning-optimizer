"""MCTS.search counts failed iterations, aborts after repeated failures, and has no
no-improvement early stop."""
import threading

import pytest

from docetl.mcts.mcts import MCTS


class _Frontier:
    frontier_plans = []

    def __len__(self):
        return 0

    def get_all_plans_summary(self):
        return []


def _bare_mcts(iteration, max_iterations=4, max_consecutive_failures=3):
    mcts = MCTS.__new__(MCTS)
    mcts.max_iterations = max_iterations
    mcts.max_concurrent_agents = 1
    mcts.tree_lock = threading.RLock()
    mcts.action_space_exhausted = False
    mcts.failed_iteration_count = 0
    mcts.consecutive_failures = 0
    mcts.max_consecutive_failures = max_consecutive_failures
    mcts.consecutive_all_skipped = 0
    mcts.max_consecutive_all_skipped = 5
    mcts.max_budget_usd = None
    mcts.total_search_cost = 0.0
    mcts.agent_cost = mcts.pipeline_cost = 0.0
    mcts.resumed = False
    mcts._checkpoint = lambda: None
    mcts.output_dir = None
    mcts.pareto_frontier = _Frontier()
    mcts.root = type("Root", (), {"cost": 0.0})()
    mcts.mcts_iteration = iteration
    return mcts


def test_aborts_after_consecutive_failures():
    def always_fails():
        raise KeyError("no_such_operator")

    mcts = _bare_mcts(always_fails)
    with pytest.raises(RuntimeError, match="3 iterations failed in a row"):
        mcts.search()
    assert mcts.failed_iteration_count == 3
    assert mcts.iteration_count == 0


def test_failures_are_counted_and_a_success_resets_the_streak():
    outcomes = iter([KeyError("x"), True, False, KeyError("y"), True, True, True])

    def iteration():
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    mcts = _bare_mcts(iteration)
    mcts.search()
    assert mcts.iteration_count == 4
    assert mcts.failed_iteration_count == 3
    assert mcts.consecutive_failures == 0


def test_no_early_stop_without_frontier_improvement():
    mcts = _bare_mcts(lambda: True, max_iterations=15)
    mcts.search()
    assert mcts.iteration_count == 15
