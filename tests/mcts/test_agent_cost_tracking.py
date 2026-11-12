"""Agent spending is counted per call, also when an instantiation fails after its retries."""
from types import SimpleNamespace

import docetl.reasoning_optimizer.directives.agent_utils as A


def test_failed_instantiation_cost_is_tracked(monkeypatch):
    monkeypatch.setattr(A, "_agent_completion", lambda *a, **k: (SimpleNamespace(), 0.25))

    def failing_instantiate():
        for _ in range(3):
            A.agent_completion(model="m", messages=[])
        raise Exception("Failed to instantiate directive after 3 attempts.")

    with A.track_agent_cost() as spent:
        try:
            failing_instantiate()
        except Exception:
            pass
    assert spent.total == 0.75
    A.agent_completion(model="m", messages=[])          # outside a tracker: nothing breaks
    with A.track_agent_cost() as outer:
        with A.track_agent_cost() as inner:
            A.agent_completion(model="m", messages=[])
        assert inner.total == 0.25
    assert outer.total == 0.25


def test_disallowed_models_are_found():
    from docetl.mcts.mcts_utils import disallowed_models
    ops = [{"name": "a", "model": "gemini/gemini-2.5-flash"}, {"name": "b", "model": "jev"},
           {"name": "c", "model": "gemini-3.1-flash-lite", "gleaning": {"model": "gemini-2.5-pro"}},
           {"name": "d", "embedding_model": "text-embedding-3-small"}]
    assert disallowed_models(ops) == ["gemini-2.5-pro", "gemini/gemini-2.5-flash", "text-embedding-3-small"]
    assert disallowed_models([{"name": "t", "embedding_model": "gemini/gemini-embedding-001"}]) == []
    assert disallowed_models(ops[1:2], default_model="gemini/gemini-3.5-flash-lite") == []
