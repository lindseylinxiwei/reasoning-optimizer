"""Regression tests for source fidelity, multilabel spans, and bounded verification."""
import re
import tempfile
import unittest
from pathlib import Path

from pipeline import collect_candidates, serialize, verify, windows
from adapters import Verifier
from evaluate import diagnostics

SCHEMA = {'A': 'First category', 'B': 'Second category'}


def offsets(text):
    return [m.span() for m in re.finditer(r'\S+', text)]


def candidate(text, start, end, label='A', key='C0', regions=None):
    return {'candidate_id': key, 'clause_type': label, 'start': start, 'end': end,
            'text': text[start:end], 'gliner_score': .4, 'source_windows': regions or [0]}


class FakeDecider:
    def __init__(self, responses, fits=True):
        self.responses, self.calls, self.allowed = iter(responses), [], fits

    def fits(self, state, questions):
        return self.allowed

    def __call__(self, state, questions):
        self.calls.append((state, questions))
        choice = next(self.responses)
        return {key: {'choice': choice} for key in questions}


class PipelineTests(unittest.TestCase):
    def test_windows_cover_unicode_and_shrink_without_losing_text(self):
        text = 'é 😀 one two\n\nthree four five six seven'
        regions = windows(text, offsets(text), 6, 2, fits=lambda t: len(t.split()) <= 3)
        covered = set()
        for r in regions:
            covered.update(range(r['start'], r['end']))
            self.assertLessEqual(len(text[r['start']:r['end']].split()), 3)
        self.assertTrue(all(i in covered for i, c in enumerate(text) if not c.isspace()))
        with self.assertRaises(ValueError):
            windows(text, offsets(text), 6, 2, fits=lambda t: False)

    def test_multilabel_duplicate_and_distinct_occurrences(self):
        text = 'Apple Apple'
        hit = {'start': 0, 'end': 5, 'text': 'Apple', 'confidence': .4}
        predictions = [{'entities': {'A': [hit], 'B': [hit]}},
                       {'entities': {'A': [{**hit, 'confidence': .8}]}},
                       {'entities': {'A': [hit]}}]
        regions = [{'start': 0, 'end': 5}, {'start': 0, 'end': 5}, {'start': 6, 'end': 11}]
        found, invalid = collect_candidates(text, regions, predictions, SCHEMA)
        self.assertEqual(len(found), 3)
        self.assertFalse(invalid)
        self.assertEqual(found[0]['source_windows'], [0, 1])
        self.assertEqual(found[0]['gliner_score'], .8)
        self.assertEqual([c['text_span'] for c in serialize(text, found, SCHEMA)], ['Apple, Apple', 'Apple'])

    def test_invalid_offsets_are_audited_and_unknown_types_fail(self):
        hit = {'start': 0, 'end': 4, 'text': 'Apple', 'confidence': .4}
        regions = [{'start': 0, 'end': 5}]
        found, invalid = collect_candidates('Apple', regions, [{'entities': {'A': [hit]}}], SCHEMA)
        self.assertEqual(found, [])
        self.assertEqual(len(invalid), 1)
        with self.assertRaises(ValueError):
            collect_candidates('Apple', regions, [{'entities': {'invented': [hit]}}], SCHEMA)

    def test_context_expands_once_and_retains_original_boundaries(self):
        text = 'left clause right'
        regions = [{'start': 0, 'end': 4}, {'start': 5, 'end': 11}, {'start': 12, 'end': 17}]
        candidates = [candidate(text, 5, 11, regions=[1])]
        decide = FakeDecider(['NEEDS_CONTEXT', 'KEEP'])
        decisions, trace = verify(text, candidates, regions, SCHEMA, decide)
        self.assertEqual(len(decide.calls), 2)
        self.assertEqual(trace[-1]['context'], [0, 17])
        self.assertEqual(serialize(text, candidates, SCHEMA, decisions),
                         [{'clause_type': 'A', 'text_span': 'clause'}, {'clause_type': 'B', 'text_span': ''}])

    def test_no_extra_call_when_context_cannot_expand(self):
        text = 'clause'
        candidates = [candidate(text, 0, 6)]
        decide = FakeDecider(['NEEDS_CONTEXT'])
        decisions, _ = verify(text, candidates, [{'start': 0, 'end': 6}], SCHEMA, decide)
        self.assertEqual(len(decide.calls), 1)
        self.assertEqual(decisions, {'C0': 'NEEDS_CONTEXT'})
        self.assertEqual(serialize(text, candidates, SCHEMA, decisions)[0]['text_span'], '')

    def test_empty_candidates_context_limit_and_failures(self):
        decide = FakeDecider([], fits=False)
        self.assertEqual(verify('x', [], [], SCHEMA, decide), ({}, []))
        decisions, trace = verify('x', [candidate('x', 0, 1)], [{'start': 0, 'end': 1}], SCHEMA, decide)
        self.assertEqual(trace[0]['status'], 'context_limit')
        self.assertEqual(decisions, {'C0': 'NEEDS_CONTEXT'})
        with self.assertRaises(ValueError):
            verify('x', [candidate('x', 0, 1)], [{'start': 0, 'end': 1}], SCHEMA, FakeDecider(['invented']))
        class Failed(FakeDecider):
            def __call__(self, *args):
                raise TimeoutError('network failure')
        with self.assertRaises(TimeoutError):
            verify('x', [candidate('x', 0, 1)], [{'start': 0, 'end': 1}], SCHEMA, Failed([]))

    def test_diagnostics_detect_false_rejections_and_missing_candidates(self):
        cs = [candidate('correct wrong', 0, 7), candidate('correct wrong', 8, 13, key='C1')]
        result = diagnostics(cs, {'A': ['correct', 'missing'], 'B': []}, {'C0': 'REJECT', 'C1': 'REJECT'})
        self.assertEqual(result['candidate_recall'], .5)
        self.assertEqual(result['verified_recall'], 0.)
        self.assertEqual(result['correct_candidates_removed'], 1)
        self.assertEqual(result['false_candidates_removed'], 1)

    def test_successful_requests_are_reused_but_changed_prompts_are_not(self):
        class Client:
            calls = 0
            def _pack(self, state, questions):
                return [questions]
            def _request(self, text, state, questions, *args):
                self.calls += 1
                return {'answers': {k: {'choice': 'KEEP'} for k in questions}}
        with tempfile.TemporaryDirectory() as folder:
            verifier = Verifier.__new__(Verifier)
            verifier.folder, verifier.client = Path(folder), Client()
            verifier('state', {'C0': {}})
            verifier('state', {'C0': {}})
            verifier('different state', {'C0': {}})
            self.assertEqual(verifier.client.calls, 2)


if __name__ == '__main__':
    unittest.main()
