import unittest
from iterative_evidence import (GROUP_SIZE, copy_units, rank_labels, select_evidence,
                                select_groups, source_units)


class IterativeEvidenceTests(unittest.TestCase):
    def test_source_coverage_abbreviations_decimals_and_table_rows(self):
        text = 'RESULTS\n\nDr. Lee found a 3.5 mg dose. No rash occurred.\n\nDrug\tFinding\nA\tFever\nB\tNone'
        units = source_units(text)
        self.assertIn('Dr. Lee found a 3.5 mg dose.', [u['text'] for u in units])
        self.assertEqual(sum(u['kind'] == 'table_row' for u in units), 3)
        self.assertTrue(all(text[u['start']:u['end']] == u['text'] for u in units))
        answers = {u['id']: {'choice': 'background'} for u in units}
        row = next(u for u in units if u['text'] == 'A\tFever')
        answers[row['id']] = {'choice': 'finding'}
        copied = copy_units(text, units, answers)
        self.assertIn('Drug\tFinding\nA\tFever', copied['evidence'])
        self.assertNotIn('B\tNone', copied['evidence'])
        with self.assertRaises(ValueError):
            copy_units(text, units, {})

    def test_every_unit_classified_and_negation_copied(self):
        text = 'RESULTS\n\nThe patient had fever. There was no rash.\n\nBACKGROUND\n\nRash is common.'
        seen = []
        def decide(state, questions):
            seen.extend(questions)
            return {k: {'choice': 'context' if 'no rash' in units[k]['text'] else
                        'finding' if 'had fever' in units[k]['text'] else 'background'} for k in questions}
        units = {u['id']: u for u in source_units(text)}
        result = select_evidence(text, decide, batch_size=2)
        self.assertEqual(seen, list(units))
        self.assertIn('There was no rash.', result['evidence'])
        self.assertNotIn('Rash is common.', result['evidence'])
        self.assertTrue(all(text[a:b] in result['evidence'] for a, b in result['source_spans']))

    def test_iterative_choices_remove_winners_and_stop_on_none(self):
        candidates = [{'id': str(i), 'label': str(i)} for i in range(3)]
        calls = []
        def decide(state, questions):
            options = questions['G000']['criteria']
            calls.append(list(options))
            choice = '1' if len(calls) == 1 else 'NONE'
            return {'G000': {'choice': choice}}
        result = rank_labels('text', 'text', candidates, decide)
        self.assertEqual(result['ranked_ids'], ['1'])
        self.assertNotIn('1', calls[1])
        self.assertEqual(len(calls), 2)

    def test_context_fallback_is_requested_once_per_group(self):
        candidates = [{'id': 'A', 'label': 'fever'}, {'id': 'B', 'label': 'rash'}]
        states = []
        def decide(state, questions):
            states.append(state)
            if len(states) == 1:
                self.assertIn('MORE_CONTEXT', questions['G000']['criteria'])
                return {'G000': {'choice': 'MORE_CONTEXT'}}
            self.assertNotIn('MORE_CONTEXT', questions['G000']['criteria'])
            return {'G000': {'choice': 'A' if len(states) == 2 else 'B'}}
        selected, trace = select_groups('full paper', 'excerpts', [candidates], decide, allow_context=True)
        self.assertEqual(selected['G000'], ['A', 'B'])
        self.assertIn('excerpts', states[0])
        self.assertTrue(all('full paper' in s for s in states[1:]))
        self.assertEqual(sum(len(t['requested_full_context']) for t in trace), 1)

    def test_large_pool_retains_ten_per_group_before_final_ranking(self):
        candidates = [{'id': str(i), 'label': str(i)} for i in range(2 * GROUP_SIZE + 1)]
        calls = []
        def decide(state, questions):
            calls.append(questions)
            self.assertTrue(all(len(q['criteria']) <= 255 for q in questions.values()))
            return {k: {'choice': next(iter(q['criteria']))} for k, q in questions.items()}
        result = rank_labels('text', 'text', candidates, decide, allow_context=True)
        self.assertEqual(len(result['survivor_ids']), 21)
        self.assertEqual(len(result['ranked_ids']), 10)
        self.assertEqual(len(set(result['ranked_ids'])), 10)
        self.assertEqual(len(calls[0]), 3)
        self.assertEqual(len(calls), 20)

    def test_invalid_choice_rejected_and_empty_pool_makes_no_call(self):
        candidates = [{'id': 'A', 'label': 'fever'}]
        with self.assertRaises(ValueError):
            rank_labels('text', 'text', candidates, lambda *_: {'G000': {'choice': 'invented'}})
        self.assertEqual(rank_labels('text', '', [], lambda *_: self.fail())['ranked_ids'], [])


if __name__ == '__main__':
    unittest.main()
