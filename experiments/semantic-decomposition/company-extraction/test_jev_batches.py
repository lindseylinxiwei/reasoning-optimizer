"""Offline HTTP-contract tests for shared-state selection and accounting."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from pipeline import JevSelector


class ByteEncoder:
    def encode(self, text, **kwargs):
        return text.encode()


class BatchTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict("os.environ", {"TYPESAFE_API_KEY": "offline-test"})
        env.start()
        self.addCleanup(env.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.selector = JevSelector.__new__(JevSelector)
        self.selector.encoder = ByteEncoder()
        self.selector.ledger_path = Path(self.tmp.name) / 'calls.jsonl'
        self.selector.max_estimated_tokens = 100000
        self.selector.max_request_tokens = 200000
        self.payloads = []

    def candidates(self, n):
        return [{'id': f'C{i:03}', 'text': f'Company {i}'} for i in range(n)]

    def invoke(self, n, decisions):
        decisions = iter(decisions)
        def post(url, **kwargs):
            payload = kwargs['json']; self.payloads.append(payload)
            answers = {name: {'type': 'choice', 'choice': next(decisions)}
                       for name in payload['questions']}
            class Response:
                status_code = 200
                def json(self):
                    return {'answers': answers, 'usage': {'input_tokens': 100, 'output_tokens': 10}}
            return Response()
        with patch('requests.post', side_effect=post):
            return self.selector('Full document', self.candidates(n))

    def test_single_question_boundary_and_usage(self):
        result = self.invoke(254, ['C253'])
        self.assertEqual(len(self.payloads), 1)
        self.assertEqual(len(self.payloads[0]['questions']['subject_company']['criteria']), 255)
        self.assertEqual(result['candidate_id'], 'C253')
        self.assertEqual(result['usage']['input_tokens'], 100)

    def test_402_candidates_shared_state_then_final(self):
        result = self.invoke(402, ['C000', 'C401', 'C401'])
        self.assertEqual(len(self.payloads), 2)
        first, final = self.payloads
        self.assertEqual(first['state'], 'Document:\nFull document')
        self.assertEqual([len(q['criteria']) for q in first['questions'].values()], [255, 149])
        self.assertEqual(set(final['questions']['subject_company']['criteria']), {'C000','C401','NONE'})
        self.assertEqual(result['candidate_id'], 'C401')
        self.assertAlmostEqual(result['cost_usd'], 200 * .042 / 1e6)
        ledger = [json.loads(x) for x in self.selector.ledger_path.read_text().splitlines()]
        self.assertEqual([x['round'] for x in ledger], [1, 2])

    def test_none_and_single_survivor_skip_final(self):
        self.assertEqual(self.invoke(255, ['NONE','NONE'])['candidate_id'], 'NONE')
        self.assertEqual(len(self.payloads), 1)
        self.payloads.clear()
        self.assertEqual(self.invoke(255, ['NONE','C254'])['candidate_id'], 'C254')
        self.assertEqual(len(self.payloads), 1)

    def test_recursive_selection(self):
        self.selector.BATCH_SIZE = 2
        result = self.invoke(5, ['C000','C002','C004', 'C000','C004', 'C004'])
        self.assertEqual(result['candidate_id'], 'C004')
        self.assertEqual(len(result['rounds']), 3)

    def test_context_packing_and_oversized_question(self):
        self.selector.max_estimated_tokens = 100
        self.selector.max_request_tokens = 35
        packed = self.selector._pack('doc', {'a': {'x': '12345'}, 'b': {'x': '12345'}})
        self.assertEqual(len(packed), 2)
        with self.assertRaises(ValueError):
            self.selector._pack('x'*100, {'a': {'x': '12345'}})

    def test_wrong_batch_answer_preserves_billed_usage(self):
        with self.assertRaises(ValueError):
            self.invoke(255, ['C254','C254'])  # C254 belongs only to batch 2.
        row = json.loads(self.selector.ledger_path.read_text())
        self.assertEqual(row['status'], 'error')
        self.assertAlmostEqual(row['cost_usd'], 100 * .042 / 1e6)

    def test_later_http_failure_retains_prior_cost(self):
        class Failure:
            status_code = 503
        original = self.selector._request
        def request(*args):
            if args[-2] == 2:
                with patch('requests.post', return_value=Failure()):
                    return original(*args)
            return original(*args)
        with patch.object(self.selector, '_request', side_effect=request):
            with self.assertRaises(RuntimeError):
                self.invoke(255, ['C000','C254'])
        rows = [json.loads(x) for x in self.selector.ledger_path.read_text().splitlines()]
        self.assertEqual([r['status'] for r in rows], ['success','error'])
        self.assertIsNotNone(rows[0]['cost_usd'])
        self.assertIsNone(rows[1]['cost_usd'])


if __name__ == '__main__':
    unittest.main()
