"""Offline contract tests: no inference credentials or paid calls required."""
import copy
import unittest
from unittest.mock import patch

from docetl.operations.filter import FilterOperation
from docetl.operations.map import MapOperation
from docetl.operations.utils.jev import build_questions, call_jev, request_jev
from docetl.runner import DSLRunner


class JevTests(unittest.TestCase):
    def runner(self):
        return DSLRunner({"default_model": "jev", "bypass_cache": True, "operations": [],
                          "pipeline": {"steps": [], "output": {"path": "/tmp/jev-test-output.json"}}}, max_threads=1)

    def test_real_filter_then_map(self):
        seen = []
        def provider(payload, timeout):
            seen.append(copy.deepcopy(payload))
            if "keep" in payload["questions"]:
                answer = {"keep": {"type": "noul", "noul": 0.5 if "solar" in payload["state"] else 0.1}}
            else:
                answer = {"sector": {"type": "choice", "choice": "energy"}}
            return {"answers": answer, "usage": {"input_tokens": 100}}
        runner = self.runner()
        filter_config = {"name": "keep_reports", "type": "filter", "model": "jev",
                         "prompt": "Keep sustainability reports: {{ input.text }}", "output": {"schema": {"keep": "boolean"}}}
        map_config = {"name": "sector", "type": "map", "model": "jev",
                      "prompt": "Choose energy or other: {{ input.text }}", "output": {"schema": {"sector": "enum[energy, other]"}}}
        records = [{"id": 1, "text": "solar"}, {"id": 2, "text": "unrelated"}]
        with patch("docetl.operations.utils.jev.request_jev", side_effect=provider), patch("docetl.operations.utils.api.completion", side_effect=AssertionError("LLM called")):
            retained, filter_cost = FilterOperation(runner, filter_config, "jev", 1).execute(records)
            output, map_cost = MapOperation(runner, map_config, "jev", 1).execute(retained)
        self.assertEqual(output, [{"id": 1, "text": "solar", "sector": "energy"}])
        self.assertEqual(records, [{"id": 1, "text": "solar"}, {"id": 2, "text": "unrelated"}])
        # The filter sends both documents concurrently, so their request order is not fixed.
        self.assertEqual({s["state"] for s in seen[:2]},
                         {"Keep sustainability reports: solar", "Keep sustainability reports: unrelated"})
        self.assertEqual(seen[-1]["questions"]["sector"]["criteria"], {"energy": None, "other": None})
        self.assertEqual(len(seen), 3)
        self.assertAlmostEqual(filter_cost + map_cost, 300 * 0.042 / 1_000_000)

    def test_invalid_schema_and_answers(self):
        for schema in ({"x": "string"}, {"x": "enum[a,a]"}, {"x": "enum[a,]"}):
            with self.assertRaises(ValueError):
                build_questions(schema)
        for value in (float("nan"), True, -0.1, 1.1, None):
            result = {"answers": {"x": {"type": "noul", "noul": value}}, "usage": {"input_tokens": 1}}
            with patch("docetl.operations.utils.jev.request_jev", return_value=result), self.assertRaises(ValueError):
                call_jev("jev", [{"role": "user", "content": "test"}], {"x": "bool"}, timeout=10, bypass_cache=True)
        with patch("docetl.operations.utils.jev.request_jev", return_value={"answers": {"x": {"type": "choice", "choice": "invalid"}}}), self.assertRaises(ValueError):
            call_jev("jev", [{"role": "user", "content": "test"}], {"x": "enum[a,b]"}, timeout=10, bypass_cache=True)

    def test_cache_bypass_and_validation(self):
        result = {"answers": {"x": {"type": "noul", "noul": 0.8}}, "usage": {"input_tokens": 10}}
        memory = {}
        with patch("docetl.operations.utils.jev.cache") as cache, patch("docetl.operations.utils.jev.request_jev", return_value=result) as request:
            cache.get.side_effect = memory.get
            cache.set.side_effect = lambda key, value: memory.__setitem__(key, value)
            def run(bypass=False, validation=None):
                return call_jev("jev", [{"role": "user", "content": "cache-test"}], {"x": "bool"}, timeout=10, bypass_cache=bypass, validation_config=validation)
            self.assertGreater(run().total_cost, 0)
            self.assertEqual(run().total_cost, 0)
            run(True)
            self.assertEqual(request.call_count, 2)
            self.assertFalse(run(validation={"validation_fn": lambda output: (output, False)}).validated)

    def test_gateway_route_and_headers(self):
        import json
        from io import BytesIO
        for base in ("https://example.invalid/.netlify/ai", "https://api.typesafe.ai/v1"):
            with patch.dict("os.environ", {"TYPESAFE_BASE_URL": base, "TYPESAFE_API_KEY": "test-key"}, clear=True), patch("docetl.operations.utils.jev.urlopen") as send:
                send.return_value.__enter__.return_value = BytesIO(json.dumps({"answers": {}}).encode())
                request_jev({"state": "test", "questions": {}}, 10)
                request = send.call_args.args[0]
                self.assertEqual(request.full_url, base + ("/systemone" if base.endswith("/v1") else "/v1/systemone"))
                self.assertEqual(request.get_header("Accept"), "application/json")
                self.assertEqual(request.get_header("Authorization"), "Bearer test-key")

    def test_unsupported_features_fail_before_inference(self):
        config = {"name": "bad", "type": "map", "model": "jev", "prompt": "{{input.text}}", "output": {"schema": {"x": "bool"}}, "calibrate": True}
        with self.assertRaises(ValueError):
            MapOperation(self.runner(), config, "jev", 1)
        with self.assertRaises(ValueError):
            self.runner().api.call_llm("jev", "reduce", [{"role": "user", "content": "x"}], {"x": "bool"})


if __name__ == "__main__":
    unittest.main()
