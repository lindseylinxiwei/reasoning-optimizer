"""Every n-th agent decision offers only the least-chosen applicable rewrites."""
import threading

from docetl.mcts.mcts import MCTS
from docetl.mcts.mcts_utils import format_action_options, rewrite_family


class Obj:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def make(attempts, every=3, k=3):
    t = Obj(attempts=[{"directive": d} for d in attempts], tree_lock=threading.RLock(),
            explore_every=every, explore_k=k)
    for name in ("_choice_counts", "_exploration_options"):
        setattr(t, name, getattr(MCTS, name).__get__(t))
    return t


OPTIONS = [("f", "take_head_tail"), ("f", "jev_classify"), ("f", "doc_chunking"), ("f", "jev_retrieval"),
           ("r", "reduce_chaining"), ("r", "change to gemini/x"), ("f", "change to gemini/y"), ("f", "gleaning")]


def test_normal_turns_are_unrestricted():
    assert make(["take_head_tail"])._exploration_options(OPTIONS) is None      # decision 2
    assert make([])._exploration_options(OPTIONS) is None                     # decision 1


def test_exploration_keeps_least_chosen_families_with_ties():
    # decision 3; counts: take_head_tail 1, jev_classify 1, everything else 0 -> all zero-count families kept
    t = make(["take_head_tail", "jev_classify"])
    kept = {n for _, n in t._exploration_options(OPTIONS)}
    assert kept == {"doc_chunking", "jev_retrieval", "reduce_chaining", "change to gemini/x",
                    "change to gemini/y", "gleaning"}


def test_exploration_takes_smallest_k_when_all_tried():
    hist = (["take_head_tail"] * 4 + ["jev_classify"] * 3 + ["doc_chunking"] * 2 + ["jev_retrieval"] +
            ["reduce_chaining"] * 3 + ["change to gemini/z"] * 2 + ["gleaning"] * 5)    # 20 -> decision 21
    kept = {n for _, n in make(hist)._exploration_options(OPTIONS)}
    # counts: jev_retrieval 1, doc_chunking 2, change model 2 (one family) -> k=3 smallest
    assert kept == {"jev_retrieval", "doc_chunking", "change to gemini/x", "change to gemini/y"}


def test_off_or_nothing_to_remove():
    assert make(["a", "b"], every=0)._exploration_options(OPTIONS) is None
    assert make(["a", "b"])._exploration_options(OPTIONS[:3]) is None


def test_options_listed_least_chosen_first():
    text = format_action_options(OPTIONS[:3], {"take_head_tail": 2})
    lines = text.splitlines()
    assert lines[-1] == "Operator: f, Rewrite directive: take_head_tail (chosen 2 times in this search)"
    assert "(not tried yet in this search)" in lines[0]
    assert rewrite_family("change to gemini/x") == "change model"
    assert format_action_options(OPTIONS[:1]) == "Operator: f, Rewrite directive: take_head_tail\n"


def test_choices_skipped_by_cost_guard_do_not_count():
    t = make([])
    t.attempts = [{"directive": "jev_retrieval", "results": [{"status": "skipped"}] * 3},
                  {"directive": "take_head_tail", "results": [{"status": "ok"}, {"status": "skipped"}]},
                  {"directive": "gleaning", "results": []}]                       # in progress
    assert t._choice_counts() == {"take_head_tail": 1, "gleaning": 1}


def test_declined_rewrites_count_as_used():
    # decision 3, nothing chosen yet; the agent declined jev_classify and take_head_tail earlier
    t = make(["gleaning", "gleaning"])
    t.declined = {"jev_classify": 3, "take_head_tail": 3, "doc_chunking": 3, "jev_retrieval": 3}
    kept = {n for _, n in t._exploration_options(OPTIONS)}
    assert kept == {"reduce_chaining", "change to gemini/x", "change to gemini/y", "gleaning"}


def test_rewrites_that_do_not_fit_the_plan_shape():
    from docetl.mcts.mcts_utils import fits_plan_structure
    one = [{"name": "extract", "type": "map"}]
    assert not fits_plan_structure(one, "extract", "chunk_header_summary")
    assert not fits_plan_structure(one, "extract", "operator_fusion")
    assert not fits_plan_structure(one, "extract", "reduce_gleaning")
    assert fits_plan_structure(one, "extract", "clarify_instructions")
    chunked = [{"name": "split", "type": "split"}, {"name": "gather", "type": "gather"},
               {"name": "m", "type": "map"}, {"name": "r", "type": "reduce"}]
    assert fits_plan_structure(chunked, "m", "chunk_header_summary")
    assert fits_plan_structure(chunked, "m", "operator_fusion")
    assert fits_plan_structure(chunked, "m", "map_reduce_fusion")
    assert fits_plan_structure(chunked, "r", "reduce_gleaning") and not fits_plan_structure(chunked, "m", "reduce_gleaning")
