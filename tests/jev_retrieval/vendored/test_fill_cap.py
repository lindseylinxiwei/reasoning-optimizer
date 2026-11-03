"""Offline synthetic tests; no model predictions or quality measurements."""
from dataclasses import replace
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import unittest

from navigation import NavigationConfig, Navigator, Task
from retrieval import PRESETS, retrieve_document
from semantic_index import IndexNode, SemanticIndex
from semantic_topic_index import build_topic_index
from test_navigation import Fixtures, small_index


ROOT = Path(__file__).resolve().parents[1]
TASK = Task("Synthetic target", "Extract synthetic provisions")


class FillCapTests(unittest.TestCase):
    def run_case(self, scores=None, index=None, **changes):
        cfg = replace(PRESETS["fill_cap_512"], workers=1, **changes)
        return Navigator(Fixtures(scores or {}), cfg).retrieve(index or small_index(), TASK)

    def assert_literal(self, result, text):
        last = 0
        total = 0
        for e in result["evidence"]:
            self.assertGreaterEqual(e["start"], last)
            self.assertEqual(e["text"], text[e["start"]:e["end"]])
            last = e["end"]
            total += len(e["text"])
        self.assertEqual(total, result["evidence_chars"])
        self.assertLessEqual(total, result["config"]["evidence_chars"])

    def test_zero_scores_fill_cap(self):
        r = self.run_case(evidence_chars=21)
        self.assertEqual(r["evidence_chars"], 21)
        self.assertTrue(r["target_filled"])
        self.assertFalse(r["score_threshold_applied"])
        self.assertEqual(r["stop_reason"], "evidence_cap_reached")
        self.assertFalse(any(c["reason"] == "below_leaf_threshold" for c in r["closures"]))
        self.assert_literal(r, small_index().text)

    def test_highest_scored_branch_selected_first(self):
        r = self.run_case({"A": .01, "B": .02}, evidence_chars=16)
        self.assertEqual([(e["start"], e["end"]) for e in r["evidence"]], [(16, 32)])

    def test_remainder_clipped_and_attributed(self):
        r = self.run_case({"A": .9, "B": .1, "A1": .7, "A2": .8}, evidence_chars=3)
        self.assertEqual(r["evidence"][0]["text"], "BBB")
        self.assertTrue(r["cap_clipped"])
        self.assertTrue(any(p["reason"] == "cap_clipped_remainder" for p in r["pending"]))
        self.assertEqual(r["unfilled_target_chars"], 0)

    def test_document_shorter_than_cap(self):
        r = self.run_case(evidence_chars=48)
        self.assertEqual(r["evidence_chars"], 32)
        self.assertEqual(r["stop_reason"], "source_exhausted")
        self.assertTrue(r["target_filled"])
        self.assertFalse(r["retrieval_partial"])

    def test_zero_cap_requires_no_score(self):
        r = self.run_case(evidence_chars=0)
        self.assertEqual(r["controller_calls"], 0)
        self.assertEqual(r["stop_reason"], "evidence_cap_reached")

    def test_zero_decisions_does_not_fabricate_evidence(self):
        r = self.run_case(max_calls=0, evidence_chars=12)
        self.assertEqual(r["evidence"], [])
        self.assertEqual(r["stop_reason"], "decision_limit")
        self.assertEqual(r["unfilled_target_chars"], 12)

    def test_already_scored_regions_admitted_at_decision_limit(self):
        r = self.run_case(max_calls=1, evidence_chars=20)
        self.assertEqual(r["evidence_chars"], 16)
        self.assertEqual(r["controller_calls"], 1)
        self.assertEqual(r["stop_reason"], "decision_limit")

    def test_increasing_limit_can_finish_same_search(self):
        text = "abcd" * 2048
        index = build_topic_index(text, [(i, i+32) for i in range(0, len(text), 32)],
                                  [[1., 0.]] * 256, fanout=2, leaf_chars=32)
        low = self.run_case(index=index, max_calls=128, admission_chars=0, evidence_chars=8001)
        high = self.run_case(index=index, max_calls=512, admission_chars=0, evidence_chars=8001)
        self.assertEqual(low["stop_reason"], "decision_limit")
        self.assertLess(low["evidence_chars"], 8001)
        self.assertGreater(high["controller_calls"], 128)
        self.assertLessEqual(high["controller_calls"], 512)
        self.assertEqual(high["evidence_chars"], 8001)
        self.assert_literal(high, text)

    def test_empty_source(self):
        index = SemanticIndex("", "R", {"R": IndexNode("R", 0, 0, "empty")})
        r = self.run_case(index=index)
        self.assertEqual(r["controller_calls"], 0)
        self.assertEqual(r["stop_reason"], "source_exhausted")
        self.assertFalse(r["retrieval_partial"])

    def test_every_cap_fills_with_literal_nonoverlapping_text(self):
        scores = {"A": .6, "B": .8, "A1": .9, "A2": .2, "B1": .1, "B2": .7}
        for cap in range(41):
            r = self.run_case(scores, evidence_chars=cap, admission_chars=0)
            self.assertEqual(r["evidence_chars"], min(cap, 32))
            self.assertTrue(r["target_filled"])
            self.assert_literal(r, small_index().text)

    def test_oversized_leaf_is_paged_then_cap_clipped(self):
        text = "契約😀abcdef" * 1000
        index = SemanticIndex(text, "R", {"R": IndexNode("R", 0, len(text), "text")})
        r = self.run_case(index=index, evidence_chars=4567, max_request_bytes=1800)
        self.assertEqual(r["evidence_chars"], 4567)
        self.assertTrue(any(e["kind"] == "capacity_partition" for e in r["trace"]))
        self.assertTrue(r["cap_clipped"])
        self.assert_literal(r, text)

    def test_question_paging_preserves_low_score_regions(self):
        r = self.run_case(evidence_chars=31, max_questions=1)
        self.assertEqual(r["evidence_chars"], 31)
        self.assertTrue(all(len(e["request"]["questions"]) <= 1 for e in r["trace"] if e.get("stage") == "request"))

    def test_unicode_cap_is_characters_not_bytes(self):
        text = "契約😀café"
        index = SemanticIndex(text, "R", {"R": IndexNode("R", 0, len(text), "text")})
        r = self.run_case(index=index, evidence_chars=3)
        self.assertEqual(r["evidence"][0]["text"], "契約😀")
        self.assert_literal(r, text)

    def test_scoring_failure_stays_partial(self):
        class Broken:
            def score(self, payload, context):
                raise RuntimeError("synthetic failure")
        r = Navigator(Broken(), PRESETS["fill_cap_512"]).retrieve(small_index(), TASK)
        self.assertEqual(r["stop_reason"], "decision_error")
        self.assertFalse(r["target_filled"])
        self.assertEqual(r["evidence"], [])

    def test_invalid_score_is_not_treated_as_zero(self):
        r = self.run_case({"A": float("nan")})
        self.assertEqual(r["stop_reason"], "decision_error")

    def test_request_capacity_failure_does_not_fill_with_unscored_text(self):
        r = self.run_case(max_request_bytes=1)
        self.assertEqual(r["evidence"], [])
        self.assertEqual(r["stop_reason"], "unresolved_source")

    def test_threshold_cannot_silently_apply_to_new_mode(self):
        with self.assertRaises(ValueError):
            NavigationConfig(variant="fill_cap", threshold=.3)
        with self.assertRaises(ValueError):
            replace(PRESETS["fill_cap_512"], adjacency_chars=1)
        with self.assertRaises(ValueError):
            replace(PRESETS["fill_cap_512"], max_calls=-1)

    def test_bundle_api_and_matched_limit_comparison(self):
        index = small_index()
        before = index.to_json()
        old = retrieve_document(index, [TASK], Fixtures({}), preset="baseline_03_128", evidence_chars=21)
        new = retrieve_document(index, [TASK], Fixtures({}), preset="fill_cap_128", evidence_chars=21)
        self.assertEqual(old["results"][TASK.name]["evidence_chars"], 0)
        self.assertEqual(new["results"][TASK.name]["evidence_chars"], 21)
        self.assertEqual(old["config"]["max_calls"], new["config"]["max_calls"])
        self.assertEqual(before, index.to_json())
        default = retrieve_document(index, [TASK], Fixtures({}))
        self.assertEqual(default["config"]["max_calls"], 512)
        override = retrieve_document(index, [TASK], Fixtures({}), max_decisions=1024)
        self.assertEqual(override["config"]["max_calls"], 1024)

    def test_concurrent_response_order_is_deterministic(self):
        cfg = replace(PRESETS["fill_cap_512"], evidence_chars=21, admission_chars=0)
        scores = {"A": .8, "B": .7, "A1": .3, "A2": .2, "B1": .4, "B2": .1}
        a = Navigator(Fixtures(scores, {"A": .01}), cfg).retrieve(small_index(), TASK)
        b = Navigator(Fixtures(scores, {"B": .01}), cfg).retrieve(small_index(), TASK)
        self.assertEqual(a["evidence"], b["evidence"])
        self.assertEqual(a["evidence_chars"], 21)

    def test_historical_baseline_bytes_unchanged(self):
        workspace = ROOT.parents[2]
        for name, expected in json.loads((ROOT / "BASELINE_HASHES.json").read_text()).items():
            self.assertEqual(hashlib.sha256((workspace / name).read_bytes()).hexdigest(), expected, name)

    def test_original_policy_outputs_match_frozen_engine(self):
        path = ROOT.parent / "jev-threshold03-full-e2e-20261028-001/implementation/navigation.py"
        spec = importlib.util.spec_from_file_location("frozen_baseline_navigation", path)
        original = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = original
        spec.loader.exec_module(original)
        def stable(value):
            if isinstance(value, dict):
                return {k: stable(v) for k, v in value.items() if k not in {"seconds", "retrieval_seconds"}}
            if isinstance(value, list):
                return [stable(v) for v in value]
            return value
        for variant in ["adaptive", "ranked_frontier", "recovery", "threshold_reference"]:
            for cap in [0, 3, 16, 21, 48]:
                for limit in [0, 1, 2, 128]:
                    kwargs = dict(variant=variant, threshold=.3, evidence_chars=cap,
                                  max_calls=limit, admission_chars=16, workers=1)
                    scores = {"A": .9, "B": .1, "A1": .05, "A2": .2, "B1": .8}
                    old = original.Navigator(Fixtures(scores), original.NavigationConfig(**kwargs)).retrieve(small_index(), TASK)
                    new = Navigator(Fixtures(scores), NavigationConfig(**kwargs)).retrieve(small_index(), TASK)
                    self.assertEqual(stable(old), stable(new), kwargs)


if __name__ == "__main__":
    unittest.main()
