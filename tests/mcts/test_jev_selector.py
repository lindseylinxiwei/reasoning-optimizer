"""Offline tests; run with python -m unittest discover -s tests/mcts -v."""
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.error import HTTPError

# The repository imports a sentiment evaluator eagerly; do not download data in tests.
with patch("nltk.download", return_value=False):
    from docetl.mcts.jev_selector import JevSelector, JevSelectionError
    from docetl.mcts.mcts import MCTS
    from docetl.reasoning_optimizer.directives.chaining import ChainingDirective


def directive(name):
    return SimpleNamespace(to_string_for_plan=lambda: f"Description of {name}")


class JevSelectorTests(unittest.TestCase):
    def setUp(self):
        self.selector = JevSelector(api_key="test-key")
        self.records = []
        self.requests = []

    def responses(self, *choices):
        choices = iter(choices)
        def request(payload):
            self.requests.append(payload)
            return {"model": "jev-1.13.0", "usage": {"input_tokens": 1000, "output_tokens": 10},
                    "answers": {"selection": {"type": "choice", "choice": next(choices),
                                "confidence": 0.8, "probabilities": {}}}}
        self.selector._request = request

    def select(self, options, ops, names=None):
        names = names or sorted({name for _, name in options})
        return self.selector.select(
            messages=[{"role": "user", "content": "existing MOAR context"}],
            action_options=options, directives={n: directive(n) for n in names},
            pipeline={"operations": ops}, goal="acc", on_call=self.records.append)

    def test_directive_and_target_choices_and_cost(self):
        self.responses("d0", "t1")
        result = self.select([("a", "chaining"), ("b", "chaining"), ("a", "gleaning")],
                             [{"name": "a", "type": "map"}, {"name": "b", "type": "map"}])
        self.assertEqual(result, {"directive": "chaining", "operators": ["b"]})
        self.assertEqual(len(self.requests), 2)
        self.assertEqual(self.requests[0]["state"]["optimization_goal"], "acc")
        self.assertAlmostEqual(sum(r["estimated_cost_usd"] for r in self.records), 0.000084)

    def test_excluded_pairs_never_reappear(self):
        self.responses("d1")
        result = self.select([("b", "gleaning"), ("a", "chaining")],
                             [{"name": "a", "type": "map"}, {"name": "b", "type": "map"}])
        self.assertEqual(result, {"directive": "gleaning", "operators": ["b"]})

    def test_required_pair(self):
        # Impossible map_reduce_fusion is eliminated before selection.
        result = self.select([("s", "chunk_header_summary"), ("g", "chunk_header_summary"),
                              ("s", "map_reduce_fusion")],
                             [{"name": "s", "type": "split"}, {"name": "g", "type": "gather"}])
        self.assertEqual(result, {"directive": "chunk_header_summary", "operators": ["s", "g"]})
        self.assertEqual(self.records, [])  # Forced choices make no paid call.

    def test_pair_obeys_pipeline_order(self):
        result = self.selector.select(messages=[],
            action_options=[("m", "map_reduce_fusion"), ("r", "map_reduce_fusion")],
            directives={"map_reduce_fusion": directive("map_reduce_fusion")},
            pipeline={"operations": [{"name": "r", "type": "reduce"}, {"name": "m", "type": "map"}],
                      "pipeline": {"steps": [{"operations": ["m", "r"]}]}},
            goal="cost", on_call=self.records.append)
        self.assertEqual(result["operators"], ["m", "r"])

    def test_multi_target_and_stop(self):
        self.responses("t0", "o1", "stop")
        result = self.select([(op, "doc_summarization") for op in "abc"],
                             [{"name": op, "type": "map"} for op in "abc"])
        self.assertEqual(result["operators"], ["a", "b"])
        self.assertEqual(len(self.records), 3)

    def test_model_change_can_target_multiple_operators(self):
        self.responses("t0", "o1")
        result = self.select([(op, "change to test-model") for op in "ab"],
                             [{"name": op, "type": "map"} for op in "ab"])
        self.assertEqual(result["operators"], ["a", "b"])

    def test_invalid_choice_still_accounts_for_paid_call(self):
        self.responses("not-a-candidate")
        with self.assertRaises(JevSelectionError):
            self.select([("a", "chaining"), ("a", "gleaning")], [{"name": "a"}])
        self.assertEqual(len(self.records), 1)

    def test_empty_and_oversized_choices_fail_without_truncation(self):
        self.selector._request = Mock(side_effect=AssertionError("must not call"))
        with self.assertRaises(JevSelectionError):
            self.select([], [])
        with self.assertRaisesRegex(JevSelectionError, "255"):
            self.selector._choose({}, {str(i): str(i) for i in range(256)}, "choose", self.records.append)

    def test_missing_key_fails_before_pipeline_execution(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "TYPESAFE_API_KEY"):
                MCTS("missing.yaml", set(), [], "", "cuad", selector_backend="jev")

    def test_http_payload_and_auth(self):
        response = Mock()
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        response.read.return_value = b'{"answers": {}, "usage": {"input_tokens": 1}}'
        with patch("docetl.mcts.jev_selector.urlopen", return_value=response) as send:
            self.selector._request({"model": "jev-1.13.0", "state": "hello", "questions": {}})
        req = send.call_args.args[0]
        self.assertEqual(req.full_url, "https://api.typesafe.ai/v1/systemone")
        self.assertEqual(req.get_header("Authorization"), "Bearer test-key")
        self.assertEqual(json.loads(req.data)["state"], "hello")

    def test_auth_error_is_not_retried_or_echoed(self):
        error = HTTPError("https://api.typesafe.ai", 401, "secret-error", {}, None)
        with patch("docetl.mcts.jev_selector.urlopen", side_effect=error) as send:
            with self.assertRaisesRegex(JevSelectionError, "HTTP 401"):
                self.selector._request({})
        self.assertEqual(send.call_count, 1)

    def test_rate_limit_retries_are_bounded(self):
        with patch("docetl.mcts.jev_selector.urlopen", side_effect=lambda *a, **k:
                   (_ for _ in ()).throw(HTTPError("url", 429, "limit", {"Retry-After": "0"}, None))) as send:
            with patch("docetl.mcts.jev_selector.time.sleep"):
                with self.assertRaisesRegex(JevSelectionError, "429"):
                    self.selector._request({})
        self.assertEqual(send.call_count, 3)


class ExpansionIntegrationTests(unittest.TestCase):
    def test_local_exhaustion_handling_preserves_fatal_jev_errors(self):
        from docetl.mcts.jev_selector import JevAPIError
        for error in (JevAPIError("HTTP 401"), RuntimeError("Action space may be exhausted")):
            with self.subTest(error=type(error).__name__):
                tree = MCTS.__new__(MCTS)
                node = SimpleNamespace(get_id=lambda: 1, visits=1, children=[], cost=1.0)
                tree.root = SimpleNamespace(children=[node], cost=-1)
                tree.tree_lock = threading.RLock()
                tree.select = Mock(return_value=node)
                tree.is_first_layer_node = Mock(return_value=False)
                tree.increment_visits_up_tree = Mock()
                tree.get_optimize_goal = Mock(return_value="acc")
                tree.expand = Mock(side_effect=error)
                tree.action_space_exhausted = False
                if isinstance(error, JevAPIError):
                    with self.assertRaises(JevAPIError):
                        tree.mcts_iteration()
                    self.assertFalse(tree.action_space_exhausted)
                else:
                    # The node is marked and skipped by selection; the search stops only when
                    # every leaf is marked (here the only one).
                    self.assertIsNone(tree.mcts_iteration())
                    self.assertTrue(node.no_actions)
                    self.assertTrue(tree.action_space_exhausted)

    def test_jev_and_llm_both_preserve_instantiation_and_history(self):
        for backend in ("jev", "llm"):
            with self.subTest(backend=backend), tempfile.TemporaryDirectory() as output:
                d = ChainingDirective()
                tree = MCTS.__new__(MCTS)  # Avoid real dataset execution in __init__.
                tree.jev_selector = Mock() if backend == "jev" else None
                tree.model = "original-instantiation-model"
                tree.tree_lock = threading.RLock()
                tree.total_search_cost = 0.0
                tree.agent_cost = tree.pipeline_cost = 0.0
                tree.iteration_count = 0
                tree.selector_calls = []
                tree.output_dir = output
                tree.costs_log_path = f"{output}/costs.jsonl"
                tree.use_param_grid = True
                tree.available_actions = {d}
                tree.directive_name_to_obj = {d.name: d}
                tree.action_counts = {d: 0}
                tree.dataset_name = "cuad"
                tree.expansion_prompt_acc = Mock(return_value=("original selection context", "condensed"))
                child = SimpleNamespace()
                tree.instantiate_node = Mock(return_value=child)
                node = SimpleNamespace(
                    op_dict={"a": {"name": "a", "type": "map"}},
                    parsed_yaml={"operations": [{"name": "a", "type": "map"}],
                                 "default_model": "pipeline-model"},
                    memo=[], value=0, used_actions={"a": set()}, message_history=[],
                    get_id=lambda: 7, mark_action_used=Mock())
                selection = {"directive": "chaining", "operators": ["a"]}
                if backend == "jev":
                    def choose(**kw):
                        kw["on_call"]({"estimated_cost_usd": 0.01})
                        return selection
                    tree.jev_selector.select.side_effect = choose
                response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(selection)))])
                def instantiate(**kw):
                    self.assertEqual(kw["agent_llm"], "original-instantiation-model")
                    self.assertEqual(kw["target_ops"], ["a"])
                    self.assertEqual(json.loads(kw["message_history"][-1]["content"]), selection)
                    return node.parsed_yaml["operations"], kw["message_history"], 0.2
                with patch.object(ChainingDirective, "instantiate", side_effect=instantiate) as inst, \
                     patch("docetl.mcts.mcts.agent_completion", return_value=(response, 0.03)) as llm:
                    self.assertEqual(tree.expand(node, "acc"), [child])
                    inst.assert_called_once()
                    self.assertEqual(llm.call_count, 0 if backend == "jev" else 1)
                self.assertAlmostEqual(tree.total_search_cost, 0.21 if backend == "jev" else 0.23)
                node.mark_action_used.assert_called_once_with("a", d)
                self.assertEqual(tree.action_counts[d], 0)    # counted once a child runs successfully
                if backend == "jev":
                    self.assertEqual(json.loads((Path(output) / "jev_selection_calls.jsonl").read_text())["node_id"], 7)


class ExperimentRunnerTests(unittest.TestCase):
    def test_runner_forwards_selector_and_saves_accounting(self):
        from experiments.reasoning import run_mcts
        fake = SimpleNamespace(
            search=Mock(return_value=[]), root=SimpleNamespace(cost=0.5),
            pareto_frontier=SimpleNamespace(plans=[], plans_accuracy={}, frontier_plans=[]),
            selector_calls=[{"estimated_cost_usd": 0.01}], total_search_cost=0.51)
        with tempfile.TemporaryDirectory() as directory:
            dataset = Path(directory) / "input.json"
            dataset.write_text("[]")
            with patch.object(run_mcts, "MCTS", return_value=fake) as constructor, \
                 patch.object(run_mcts, "get_dataset_stats", return_value="test stats"), \
                 patch.object(run_mcts, "run_dataset_evaluation", return_value=(None, None)):
                result = run_mcts.run_mcts_experiment(
                    "unused.yaml", str(dataset), output_dir=directory,
                    experiment_name="jev_test", selector_backend="jev",
                    jev_model="jev-test", jev_input_price=0.123,
                    model="instantiation-model")
            self.assertEqual(constructor.call_args.kwargs["selector_backend"], "jev")
            self.assertEqual(constructor.call_args.kwargs["jev_model"], "jev-test")
            self.assertEqual(constructor.call_args.kwargs["model"], "instantiation-model")
            saved = json.loads((Path(directory) / "jev_test" / "experiment_summary.json").read_text())
            self.assertEqual(saved["jev_selection_cost"], 0.01)
            self.assertEqual(saved["total_search_cost"], 0.51)
            self.assertEqual(result["jev_selection_calls"], 1)


if __name__ == "__main__":
    unittest.main()
