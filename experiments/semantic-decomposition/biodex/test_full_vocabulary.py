import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core import label_questions
from full_vocabulary import assemble, call_metrics, label_diagnostics, request_batch


class FakeClient:
    def __init__(self, statuses):
        self.statuses = iter(statuses)
        self.calls = 0

    def _request(self, text, state, questions, *_):
        status = next(self.statuses)
        self.calls += 1
        answers = {k: {'type': 'choice', 'choice': 'supported',
                       'probabilities': {'supported': .9, 'not_supported': .1}} for k in questions}
        body = {'answers': answers, 'usage': {'input_tokens': 10}}
        record = {'request': {'state': state, 'questions': questions}, 'http_status': status,
                  'response': body, 'status': 'success' if status == 200 else 'error',
                  'cost_usd': .01 if status == 200 else None, 'wall_seconds': .1}
        self.ledger_path.write_text(json.dumps(record) + '\n')
        if status != 200:
            raise RuntimeError('HTTP failure')
        return body


class FullVocabularyTests(unittest.TestCase):
    def test_multiple_labels_ties_and_complete_coverage(self):
        candidates = [{'id': 'B', 'label': 'rash'}, {'id': 'A', 'label': 'fever'}, {'id': 'C', 'label': 'other'}]
        answers = {k: {'choice': 'supported', 'probabilities': {'supported': .9}} for k in ['B', 'A']}
        answers['C'] = {'choice': 'not_supported', 'probabilities': {'supported': .1}}
        self.assertEqual(assemble(candidates, answers)[0], ['B', 'A'])
        with self.assertRaises(ValueError):
            assemble(candidates, {'B': answers['B']})
        self.assertEqual(len(label_questions(candidates)), 3)
        self.assertTrue(all(len(q['criteria']) == 2 for q in label_questions(candidates).values()))

    def test_transient_retry_logged_and_resume_makes_no_call(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp) / 'batch-0001'
            client = FakeClient([520, 200])
            questions = label_questions([{'id': 'A', 'label': 'rash'}])
            with patch('full_vocabulary.time.sleep'):
                first = request_batch(client, 'paper', 'state', questions, folder)
            self.assertEqual(client.calls, 2)
            self.assertEqual(request_batch(client, 'paper', 'state', questions, folder), first)
            self.assertEqual(client.calls, 2)
            stats = call_metrics(Path(temp))
            self.assertEqual(stats['unknown_cost_calls'], 1)
            self.assertEqual(stats['known_api_cost_usd'], .01)
            self.assertEqual(stats['retry_calls'], 1)
            with self.assertRaises(ValueError):
                request_batch(client, 'paper', 'changed state', questions, folder)

    def test_non_transient_is_not_retried(self):
        with tempfile.TemporaryDirectory() as temp:
            client = FakeClient([400])
            with self.assertRaises(RuntimeError):
                request_batch(client, 'paper', 'state', {'A': {}}, Path(temp))
            self.assertEqual(client.calls, 1)

    def test_recovers_response_saved_before_checkpoint(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            client = FakeClient([200])
            client.ledger_path = folder / 'attempt-01.jsonl'
            response = client._request('paper', 'state', {'A': {}}, 1, 1)
            self.assertEqual(request_batch(client, 'paper', 'state', {'A': {}}, folder), response['answers'])
            self.assertEqual(client.calls, 1)

    def test_diagnostics_distinguish_missing_label_rejection_and_ranking(self):
        candidates = [{'id': 'A', 'label': 'rash'}, {'id': 'B', 'label': 'fever'}]
        answers = {'A': {'choice': 'supported', 'probabilities': {'supported': 1}},
                   'B': {'choice': 'not_supported', 'probabilities': {'supported': .2}}}
        d = label_diagnostics(candidates, answers, ['Rash', 'fever', 'absent'])
        self.assertEqual(d['vocabulary_recall'], 2 / 3)
        self.assertEqual(d['supported_ground_truth_recall'], 1 / 3)
        self.assertEqual(d['supported_labels'], 1)


if __name__ == '__main__':
    unittest.main()
