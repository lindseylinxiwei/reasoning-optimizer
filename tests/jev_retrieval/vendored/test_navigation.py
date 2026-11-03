"""Synthetic controller fixtures only; these are never model predictions."""
from dataclasses import replace
import json
import threading
import time
import unittest

from navigation import NavigationConfig, Navigator, Task, merged_intervals
from semantic_index import IndexNode, SemanticIndex, build_index


def small_index():
    text = "AAAAaaaaBBBBbbbbCCCCccccDDDDdddd"
    nodes = {
        "R": IndexNode("R", 0, 32, "root", ("A", "B")),
        "A": IndexNode("A", 0, 16, "section A", ("A1", "A2"), "R"),
        "B": IndexNode("B", 16, 32, "section B", ("B1", "B2"), "R"),
        "A1": IndexNode("A1", 0, 8, "first", (), "A"),
        "A2": IndexNode("A2", 8, 16, "second", (), "A"),
        "B1": IndexNode("B1", 16, 24, "third", (), "B"),
        "B2": IndexNode("B2", 24, 32, "fourth", (), "B"),
    }
    return SemanticIndex(text, "R", nodes)


class Fixtures:
    kind = "offline_scripted_not_model_predictions"
    supports_concurrent_calls = True

    def __init__(self, scores, delays=None):
        self.scores = scores
        self.delays = delays or {}
        self.calls = []
        self.completion_order = []
        self.lock = threading.Lock()

    def score(self, payload, context):
        with self.lock:
            self.calls.append((payload, context))
        time.sleep(self.delays.get(context["node_id"], 0))
        result = {label: self.scores.get(region["node_id"], 0.0)
                  for label, region in context["regions"].items()}
        with self.lock:
            self.completion_order.append(context["node_id"])
        return result


class NavigationTests(unittest.TestCase):
    task = Task("Synthetic target", "Extract every synthetic matching provision.")

    def run_fixture(self, scores, **changes):
        config = replace(NavigationConfig(), **changes)
        return Navigator(Fixtures(scores), config).retrieve(small_index(), self.task)

    def test_deferred_weak_branch_remains_recoverable(self):
        scores = {"A": .9, "B": .1, "A1": .1, "A2": .1, "B1": .9, "B2": .1}
        full = self.run_fixture(scores, workers=1)
        self.assertEqual([(e["start"], e["end"]) for e in full["evidence"]], [(16, 24)])
        bounded = self.run_fixture(scores, workers=1, max_calls=2)
        self.assertEqual(bounded["retrieval_status"], "partial")
        self.assertTrue(any(p["start"] == 16 and p["end"] == 32 for p in bounded["pending"]))
        reference = self.run_fixture(scores, workers=1, variant="threshold_reference")
        self.assertEqual(reference["evidence"], [])
        self.assertEqual(reference["decision_calls"], 2)

    def test_call_limit_is_finite_even_with_positive_tree(self):
        result = self.run_fixture({k: .8 for k in small_index().nodes}, max_calls=1)
        self.assertEqual(result["decision_calls"], 1)
        self.assertTrue(result["retrieval_partial"])
        self.assertTrue(result["pending"])

    def test_zero_budget_does_not_claim_complete_absence(self):
        result = self.run_fixture({}, max_calls=0)
        self.assertEqual(result["decision_calls"], 0)
        self.assertTrue(result["retrieval_partial"])
        self.assertTrue(result["empty_evidence_does_not_prove_absence"])

    def test_scored_leaves_admitted_at_call_limit(self):
        result = self.run_fixture({"A": .9, "B": .1, "A1": .9, "A2": .8}, max_calls=2, workers=1)
        self.assertEqual(result["evidence_chars"], 16)
        self.assertEqual(result["decision_calls"], 2)
        self.assertTrue(result["retrieval_partial"])

    def test_adaptive_admits_complete_internal_node(self):
        scores = {"A": .9, "B": .1, "B1": 0, "B2": 0}
        result = self.run_fixture(scores, variant="adaptive", admission_chars=16)
        self.assertEqual(result["evidence_chars"], 16)
        self.assertEqual(result["decision_calls"], 2)
        self.assertTrue(any(e.get("reason") == "adaptive_internal_region" for e in result["trace"]))

    def test_parent_child_disagreement_recovery(self):
        scores = {"A": .9, "B": .1, "A1": .12, "A2": .13}
        result = self.run_fixture(scores, variant="recovery", admission_chars=0, recovery_chars=16)
        self.assertEqual(result["evidence_chars"], 16)
        self.assertEqual(result["recoveries"], 1)
        self.assertTrue(any(e["kind"] == "parent_child_disagreement" for e in result["trace"]))

    def test_recovery_limit_applies_to_concurrently_queued_recoveries(self):
        scores = {"A": .9, "B": .8, "A1": .1, "A2": .1, "B1": .1, "B2": .1}
        result = self.run_fixture(scores, variant="recovery", admission_chars=0, max_recoveries=1)
        self.assertEqual(result["recoveries"], 1)
        self.assertEqual(result["evidence_chars"], 16)

    def test_recovery_never_admits_over_cap_parent(self):
        result = self.run_fixture({"A": .9, "B": .1}, variant="recovery", admission_chars=0, evidence_chars=8)
        self.assertEqual(result["evidence"], [])
        self.assertTrue(any(e["kind"] == "recovery_cannot_fit" for e in result["trace"]))

    def test_unique_source_cap_and_partial_overflow(self):
        result = self.run_fixture({k: .9 for k in small_index().nodes}, evidence_chars=12)
        self.assertLessEqual(result["evidence_chars"], 12)
        self.assertEqual(result["evidence_chars"], 8)
        self.assertTrue(result["retrieval_partial"])
        self.assertTrue(any(i["reason"] == "evidence_limit" for i in result["issues"]))

    def test_union_preserves_exact_source_and_deduplicates_overlap(self):
        self.assertEqual(merged_intervals([(0, 10), (8, 15), (15, 20), (25, 30)]), [(0, 20), (25, 30)])
        result = self.run_fixture({k: .9 for k in small_index().nodes})
        self.assertEqual(result["evidence_chars"], 32)
        self.assertEqual(result["evidence"][0]["text"], small_index().text)

    def test_all_negative_returns_empty_without_forced_answer(self):
        result = self.run_fixture({})
        self.assertEqual(result["evidence"], [])
        self.assertEqual(result["retrieval_status"], "model_search_finished")
        self.assertEqual(result["decision_calls"], 3)
        self.assertTrue(result["empty_evidence_does_not_prove_absence"])

    def test_same_wave_reversed_finish_order_has_identical_schedule(self):
        scores = {"A": .9, "B": .8, "A1": .9, "A2": .1, "B1": .8, "B2": .1}
        first = Fixtures(scores, {"A": .04})
        second = Fixtures(scores, {"B": .04})
        config = NavigationConfig(workers=4, evidence_chars=8)
        a = Navigator(first, config).retrieve(small_index(), self.task)
        b = Navigator(second, config).retrieve(small_index(), self.task)
        self.assertNotEqual(first.completion_order, second.completion_order)
        self.assertEqual(a["evidence"], b["evidence"])
        self.assertEqual(a["pending"], b["pending"])
        self.assertEqual(a["issues"], b["issues"])
        self.assertEqual([e["request_sha256"] for e in a["trace"] if e.get("stage") == "request"],
                         [e["request_sha256"] for e in b["trace"] if e.get("stage") == "request"])

    def test_full_text_markers_and_sidecar_mapping(self):
        scorer = Fixtures({})
        Navigator(scorer).retrieve(small_index(), self.task)
        payload, context = scorer.calls[0]
        self.assertIn("⟦REGION|BEGIN C0⟧\nAAAAaaaaBBBBbbbb\n⟦REGION|END C0⟧", payload["state"]["source_text"])
        self.assertIn("CCCCccccDDDDdddd", payload["state"]["source_text"])
        self.assertEqual(context["regions"]["C0"]["node_id"], "A")
        self.assertNotIn("candidates", payload["state"])
        self.assertNotIn("selected_evidence", payload["state"])

    def test_capacity_pages_preserve_every_utf8_source_character(self):
        text = "甲乙丙丁" * 3000
        index = SemanticIndex(text, "R", {"R": IndexNode("R", 0, len(text), "root")})
        scorer = Fixtures({})
        result = Navigator(scorer, NavigationConfig(max_calls=32, max_request_bytes=3000)).retrieve(index, self.task)
        intervals = sorted((r["start"], r["end"]) for _, c in scorer.calls for r in c["regions"].values())
        self.assertEqual(merged_intervals(intervals), [(0, len(text))])
        self.assertEqual(sum(end - start for start, end in intervals), len(text))
        for payload, context in scorer.calls:
            self.assertLessEqual(len(json.dumps(payload, ensure_ascii=False).encode()), 3000)
            self.assertLessEqual(len(payload["questions"]), 32)
            for label, region in context["regions"].items():
                self.assertIn(text[region["start"]:region["end"]], payload["state"]["source_text"])
        self.assertFalse(result["retrieval_partial"])

    def test_question_limit_pages_every_child(self):
        text = "A" * 80
        nodes = {"R": IndexNode("R", 0, 80, "root", tuple(f"N{i}" for i in range(40)))}
        nodes.update({f"N{i}": IndexNode(f"N{i}", 2*i, 2*i+2, "", (), "R") for i in range(40)})
        scorer = Fixtures({})
        result = Navigator(scorer, NavigationConfig(max_questions=4)).retrieve(SemanticIndex(text, "R", nodes), self.task)
        self.assertEqual(result["decision_calls"], 10)
        self.assertEqual(result["presence_question_attempts"], 40)
        self.assertTrue(all(len(p["questions"]) <= 4 for p, _ in scorer.calls))

    def test_provider_failure_drains_wave_without_replacements(self):
        class Failed(Fixtures):
            def score(self, payload, context):
                if context["node_id"] == "A":
                    raise RuntimeError("synthetic failure")
                return super().score(payload, context)
        result = Navigator(Failed({"A": .9, "B": .8, "B1": .9})).retrieve(small_index(), self.task)
        self.assertEqual(result["decision_calls"], 3)
        self.assertTrue(result["retrieval_partial"])
        self.assertEqual(sum(e.get("stage") == "failure" for e in result["trace"]), 1)
        self.assertEqual(sum(e.get("stage") == "response" for e in result["trace"]), 2)

    def test_raw_jev_response_and_invalid_scores(self):
        def raw(payload):
            return {"model": payload["model"], "answers": {label: {"type": "noul", "noul": 0.0}
                                                               for label in payload["questions"]}}
        result = Navigator(raw).retrieve(small_index(), self.task)
        self.assertFalse(result["retrieval_partial"])
        result = Navigator(lambda p: {label: float("nan") for label in p["questions"]}).retrieve(small_index(), self.task)
        self.assertTrue(result["retrieval_partial"])
        self.assertEqual(result["issues"][0]["reason"], "decision_error")

    def test_unreachable_source_is_rejected_before_any_provider_call(self):
        index = small_index()
        index.nodes["A1"] = replace(index.nodes["A1"], start=1)
        with self.assertRaises(ValueError):
            Navigator(Fixtures({})).retrieve(index, self.task)

    def test_real_index_builder_integration(self):
        text = "ARTICLE I INTRODUCTION\n\n1.1 First. A provision.\n\n1.2 Second. Another provision.\n\nARTICLE II OTHER\n\n2.1 Third. Text."
        index = build_index(text, leaf_chars=32)
        scorer = Fixtures({key: .9 for key in index.nodes})
        result = Navigator(scorer).retrieve(index, self.task)
        self.assertEqual(result["evidence_chars"], len(text))
        self.assertEqual(result["evidence"][0]["text"], text)
        self.assertFalse(result["retrieval_partial"])


if __name__ == "__main__":
    unittest.main()
