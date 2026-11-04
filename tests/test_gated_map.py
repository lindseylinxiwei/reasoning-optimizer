"""gated_map runs the map only on documents whose gate field is true; the rest get None outputs."""
from docetl.operations import get_operation
from docetl.operations.gated_map import GatedMapOperation
from docetl.operations.map import MapOperation


def test_gated_map_is_registered():
    assert get_operation("gated_map") is GatedMapOperation


def test_gated_map_only_runs_where_the_gate_is_true(monkeypatch):
    seen = []

    def fake_execute(self, items):
        seen.extend(i["id"] for i in items)
        return [{**i, "answer": "x"} for i in items], 0.5

    monkeypatch.setattr(MapOperation, "execute", fake_execute)
    op = GatedMapOperation.__new__(GatedMapOperation)
    op.config = {"gate_key": "present", "output": {"schema": {"answer": "string"}}}
    out, cost = op.execute([{"id": 1, "present": True}, {"id": 2, "present": False}, {"id": 3, "present": True}])
    assert seen == [1, 3] and cost == 0.5
    assert [o["answer"] for o in out] == ["x", None, "x"] and [o["id"] for o in out] == [1, 2, 3]


def test_gated_map_with_no_open_gates_makes_no_call(monkeypatch):
    monkeypatch.setattr(MapOperation, "execute", lambda self, items: (_ for _ in ()).throw(AssertionError("called")))
    op = GatedMapOperation.__new__(GatedMapOperation)
    op.config = {"gate_key": "present", "output": {"schema": {"answer": "string"}}}
    out, cost = op.execute([{"id": 1, "present": False}])
    assert out == [{"id": 1, "present": False, "answer": None}] and cost == 0.0
