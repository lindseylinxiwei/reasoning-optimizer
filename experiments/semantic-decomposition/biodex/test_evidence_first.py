import unittest
from core import digest, label_questions
from iterative_evidence import copy_units, source_units
from evidence_first import scan_labels, validate_evidence


class EvidenceFirstTests(unittest.TestCase):
    def test_scan_changes_only_context_and_covers_all_labels(self):
        candidates = [{'id': f'L{i:05d}', 'label': f'label {i}'} for i in range(260)]
        seen, states = {}, []
        def decide(state, questions):
            self.assertLessEqual(len(questions), 256)
            self.assertFalse(set(seen) & set(questions))
            states.append(state)
            seen.update(questions)
            return {k: {'choice': 'supported' if k == 'L00259' else 'not_supported',
                        'probabilities': {'supported': 1. if k == 'L00259' else 0.}} for k in questions}
        result = scan_labels('selected evidence', candidates, decide)
        self.assertEqual(seen, label_questions(candidates))
        self.assertEqual(states, ['Medical paper:\nselected evidence'] * 2)
        self.assertEqual(result['supported_ids'], ['L00259'])
        self.assertEqual(result['probability_ranked_ids'], ['L00259'])

    def test_missing_scan_decision_fails(self):
        with self.assertRaises(ValueError):
            scan_labels('evidence', [{'id': 'A', 'label': 'fever'}], lambda *_: {})

    def test_empty_evidence_abstains_without_calls(self):
        result = scan_labels('', [{'id': 'A', 'label': 'fever'}], lambda *_: self.fail())
        self.assertTrue(result['empty_evidence'])
        self.assertEqual(result['supported_ids'], [])

    def test_cached_evidence_source_and_copy_integrity(self):
        text = 'RESULTS\n\nFever occurred. No rash occurred.'
        units = source_units(text)
        answers = {u['id']: {'choice': 'finding'} for u in units}
        result = {**copy_units(text, units, answers), 'units': units, 'answers': answers,
                  'source_sha256': digest(text), 'source_characters': len(text)}
        self.assertIs(validate_evidence(text, result), result)
        with self.assertRaises(ValueError):
            validate_evidence(text + 'different', result)
        with self.assertRaises(ValueError):
            validate_evidence(text, {**result, 'evidence': 'invented text'})


if __name__ == '__main__':
    unittest.main()
