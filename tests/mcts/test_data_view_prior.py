"""Per-operation head cuts, the sweep prior in the prompt order, and the head-cut cost estimate."""
import threading
from types import SimpleNamespace

import yaml

from docetl.mcts.mcts import MCTS
from docetl.mcts.mcts_utils import format_action_options
from docetl.reasoning_optimizer.directives.take_head_tail import TakeHeadTailDirective
from docetl.reasoning_optimizer.instantiate_schemas import TakeHeadTailInstantiateSchema as Cut

OPS = yaml.safe_load(open("experiments/reasoning/pipelines/sustainability.yaml"))["operations"]
D = TakeHeadTailDirective()


def cut(ops, op, words, key="tot_text_raw"):
    return D.apply(ops, op, Cut(name=f"cut_{op}", document_key=key, head_words=words))


def reads(ops, name):
    p = next(o for o in ops if o["name"] == name)["prompt"]
    return [k for k in ("tot_text_raw_for_" + name, "tot_text_raw") if f".{k}" in p][0]


def test_cut_is_per_operation():
    ops = cut(OPS, "determine_economic_activity", 500)
    assert reads(ops, "determine_economic_activity") == "tot_text_raw_for_determine_economic_activity"
    assert reads(ops, "filter_sustainability_reports") == "tot_text_raw"
    assert reads(ops, "sustainability_summary") == "tot_text_raw"         # reduce reads item.tot_text_raw


def test_second_cut_replaces_the_first():
    ops = cut(cut(OPS, "sustainability_summary", 500), "sustainability_summary", 2000)
    codes = [o["code"] for o in ops if o["type"] == "code_map"]
    assert len(codes) == 1 and "words[:2000]" in codes[0]


def test_prior_counts_order_and_label():
    opts = [("f", "take_head_tail"), ("f", "gleaning"), ("m", "jev_retrieval")]
    lines = format_action_options(opts, {"gleaning": 2}, {"take_head_tail": 5, "jev_retrieval": 6}).splitlines()
    assert "gleaning" in lines[0]
    assert lines[1].endswith("(5 configurations tested in the earlier data-view sweep; not tried yet in this search)")


def test_exploration_ranks_with_prior():
    t = SimpleNamespace(attempts=[{"directive": "gleaning"}, {"directive": "chaining"}], tree_lock=threading.RLock(),
                        explore_every=3, explore_k=1, prior_counts={"take_head_tail": 5})
    t._choice_counts = MCTS._choice_counts.__get__(t)
    opts = [("f", "take_head_tail"), ("f", "gleaning"), ("f", "chaining"), ("f", "presence_gate")]
    assert MCTS._exploration_options(t, opts) == [("f", "presence_gate")]


def test_head_cut_estimate_only_when_every_reader_is_cut():
    partial = SimpleNamespace(parsed_yaml={"operations": cut(OPS, "filter_sustainability_reports", 500)})
    assert MCTS._head_cut_estimate(partial, 12.31) is None
    ops = OPS
    for name in ("filter_sustainability_reports", "determine_economic_activity", "sustainability_summary"):
        ops = cut(ops, name, 500)
    est = MCTS._head_cut_estimate(SimpleNamespace(parsed_yaml={"operations": ops}), 12.31)
    assert 0.3 < est < 1.0         # 2 x (500 words x 1.39 / 33k tokens) x $12.31
