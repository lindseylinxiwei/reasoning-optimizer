"""The readable tree log."""
from docetl.mcts.tree_log import render_tree


def test_tree_lines_frontier_and_skips():
    nodes = [
        {"id": 0, "parent": None, "action": None, "visits": 9},
        {"id": 1, "parent": 0, "action": "change to gemini-3.1-flash-lite", "cost": 1.6, "acc": 0.72, "visits": 3},
        {"id": 7, "parent": 1, "action": "map_reduce_fusion", "targets": ["m", "r"], "goal": "acc",
         "cost": 1.1, "acc": 0.76, "visits": 1},
        {"id": 8, "parent": 1, "action": "take_head_tail", "targets": ["f"], "goal": "cost",
         "grid": {"param": "head_words", "value": 1000}, "cost": 2.0, "acc": 0.70, "visits": 1},
    ]
    events = [{"id": 9, "parent": 1, "status": "skipped", "reason": "estimated $5 is more than 10% of the remaining $9"},
              {"id": 10, "parent": 1, "status": "failed"}]
    text = render_tree(nodes, events, "hdr", new_ids={8})
    lines = text.splitlines()
    assert lines[0] == "hdr"
    row7 = next(l for l in lines if "#7 " in l)
    assert "map_reduce_fusion(m,r)" in row7 and "acc" in row7 and "$1.10" in row7 and "(+0.040)" in row7
    assert row7.endswith("★")
    row8 = next(l for l in lines if "#8 " in l)
    assert "head_words=1000" in row8 and "★" not in row8 and row8.endswith("+")
    assert "model → 3.1-flash-lite" in next(l for l in lines if "#1 " in l)
    assert lines[-1].strip().endswith("(1 failed, 1 skipped: cost guard)")
