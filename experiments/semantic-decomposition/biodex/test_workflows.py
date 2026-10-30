import tempfile
import unittest
from pathlib import Path

import numpy as np

from hierarchy import build_hierarchy, validate_hierarchy
from workflows import copy_passages, search_hierarchy, select_evidence


def answer(choice, **probabilities):
    return {'type': 'choice', 'choice': choice, 'probabilities': probabilities}


class WorkflowTests(unittest.TestCase):
    def test_copy_merges_overlap_and_preserves_unicode(self):
        text = 'patient 患者 rash and fever'
        chunks = [{'start': 0, 'end': 15, 'text': text[:15]},
                  {'start': 11, 'end': len(text), 'text': text[11:]}]
        copied, spans = copy_passages(text, chunks, [0, 1])
        self.assertEqual(copied, text)
        self.assertEqual(spans, [[0, len(text)]])
        chunks[0]['text'] = 'corrupted'
        with self.assertRaises(ValueError):
            copy_passages(text, chunks, [0])

    def test_evidence_keeps_uncertainty_and_never_receives_ground_truth(self):
        text = 'Patient rash. Background nausea. Unclear fever.'
        row = {'evidence': text, 'ground_truth_reactions': ['SECRET'], 'passages': [
            {'start': a, 'end': b, 'text': text[a:b]} for a, b in [(0, 13), (14, 32), (33, 47)]]}
        def decide(state, questions):
            self.assertNotIn('SECRET', str((state, questions)))
            return {k: answer(c, **{c: 1.}) for k, c in
                    zip(questions, ['reported', 'background', 'uncertain'])}
        result = select_evidence(row, decide)
        self.assertEqual(result['selected_passage_indices'], [0, 2])
        self.assertNotIn('Background', result['evidence'])

    def test_search_preserves_multiple_leaves_and_all_labels_inside_them(self):
        labels = ['rash', 'fever', 'nausea', 'vomiting']
        tree = build_hierarchy(labels, np.array([[1, 0], [1, .01], [0, 1], [.01, 1]]),
                               leaf_size=2, branching=2, representatives=2)
        validate_hierarchy(tree, 4)
        def decide(state, questions):
            return {k: answer('supported', supported=.8, uncertain=.1, none=.1) for k in questions}
        candidates, details = search_hierarchy('paper', labels, tree, decide, beam=2)
        self.assertEqual([c['label'] for c in candidates], labels)
        self.assertEqual(len(details['reached_leaves']), 2)

    def test_search_can_abstain_without_dense_fallback(self):
        labels = ['a', 'b', 'c']
        tree = build_hierarchy(labels, np.eye(3), leaf_size=1, branching=2, representatives=1)
        def decide(state, questions):
            return {k: answer('none', none=1.) for k in questions}
        candidates, details = search_hierarchy('paper', labels, tree, decide, beam=2)
        self.assertEqual(candidates, [])
        self.assertEqual(len(details['rounds']), 1)


if __name__ == '__main__':
    unittest.main()
