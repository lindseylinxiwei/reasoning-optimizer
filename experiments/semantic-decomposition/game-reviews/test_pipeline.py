import itertools
import json
import tempfile
import unittest
from pathlib import Path

from pipeline import evenly_spaced, parse_reviews, passages, seconds, select_reviews, summarize
from run import Choices, metrics
from evaluate import audit_outputs


def review(i, body='Good game.', day=None):
    return {'review_id': str(i), 'review_text': body, 'timestamp': f'2020-01-{day or i:02} 00:00:00'}


class PipelineTests(unittest.TestCase):
    def test_complete_multiline_parsing_and_failure(self):
        text = 'Review ID: 1\nReview: Good.\nBut crashes.\nHelpful Votes: 2\nTimestamp: 2020-01-01 00:00:00'
        self.assertEqual(parse_reviews(text), [review(1, 'Good.\nBut crashes.')])
        self.assertEqual(parse_reviews(text + '\n\n' + text), [review(1, 'Good.\nBut crashes.')])
        conflicting = text + '\n\n' + text.replace('Good.', 'Bad.')
        for invalid in (text + '\nunknown data', conflicting, text.replace('2020-01-01', '2020-13-01')):
            with self.assertRaises(ValueError):
                parse_reviews(invalid)

    def test_temporal_assignment_matches_exhaustive_optimum(self):
        reviews = [review(i) for i in (1, 2, 3, 8, 9, 10)]
        start, end = seconds(reviews[0]['timestamp']), seconds(reviews[-1]['timestamp'])
        selected = evenly_spaced(reviews, start, end, 3)
        targets = [start, (start + end) / 2, end]
        cost = lambda group: sum(abs(seconds(r['timestamp']) - t) for r, t in zip(group, targets))
        self.assertEqual(cost(selected), min(cost(c) for c in itertools.combinations(reviews, 3)))
        self.assertEqual(len({r['review_id'] for r in selected}), 3)
        self.assertEqual(evenly_spaced(reviews[:2], start, end, 10), reviews[:2])

    def test_neutral_is_not_used_to_fill_a_shortage(self):
        reviews = [review(i) for i in (1, 2, 3)]
        answers = {f'R{i}': {'choice': label} for i, label in enumerate(('POSITIVE', 'NEGATIVE', 'NEUTRAL'), 1)}
        result = select_reviews(reviews, answers)
        self.assertEqual([r['review_id'] for r in result['POSITIVE']], ['1'])
        self.assertEqual([r['review_id'] for r in result['NEGATIVE']], ['2'])
        with self.assertRaises(ValueError):
            select_reviews(reviews, {'R1': {'choice': 'POSITIVE'}})

    def test_passages_preserve_source_and_merge_without_losing_words(self):
        text = '  Fine game!\nSome bugs. Still fun.\n'
        chunks = passages(text, limit=2)
        self.assertLessEqual(len(chunks), 2)
        self.assertEqual(' '.join(' '.join(c['text'].split()) for c in chunks), ' '.join(text.split()))
        for c in chunks:
            self.assertEqual(text[c['start']:c['end']], c['text'])

    def test_summary_choice_copies_source_and_singletons_skip_model(self):
        def decide(state, questions):
            self.assertEqual(list(questions), ['S1'])
            return {'S1': {'choice': 'P001'}}
        result, trace = summarize({'POSITIVE': [review(1, 'Slow start. Great ending!')],
                                   'NEGATIVE': [review(2, 'Bad.')]}, decide, 'policy')
        self.assertEqual(result['POSITIVE'][0]['review_summary'], 'Great ending!')
        self.assertEqual(trace['2']['candidate_count'], 1)

    def test_choice_cache_and_token_packing(self):
        class Client:
            calls = 0
            def _pack(self, state, questions):
                return [questions]
            def _request(self, document, state, questions, *args):
                self.calls += 1
                return {'answers': {k: {'choice': 'YES'} for k in questions}}
        with tempfile.TemporaryDirectory() as folder:
            client = Choices.__new__(Choices)
            client.folder, client.document, client.client = Path(folder), 'document', Client()
            questions = {str(i): {'criteria': {'YES': ''}} for i in range(130)}
            self.assertEqual(len(client('state', questions)), 130)
            self.assertEqual(client.client.calls, 3)
            client('state', questions)
            self.assertEqual(client.client.calls, 3)

    def test_usage_ledger_preserves_unicode_line_separators(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / '1/requests/sentiment/calls.jsonl'
            path.parent.mkdir(parents=True)
            entry = {'request': {'text': 'one\u2028two\u0085three'}, 'cost_usd': .042,
                     'response': {'usage': {'input_tokens': 1_000_000}}, 'wall_seconds': 1.}
            path.write_text(json.dumps(entry, ensure_ascii=False) + '\n')
            result = metrics(folder)
            self.assertEqual(result['api_calls'], 1)
            self.assertEqual(result['known_api_cost_usd'], .042)
            self.assertEqual(result['input_tokens'], 1_000_000)

    def test_requested_slot_score_penalizes_duplicates_and_missing_reviews(self):
        source = 'Review ID: 1\nReview: Good.\nHelpful Votes: 0\nTimestamp: 2020-01-01 00:00:00'
        row = {'app_id': 1, 'app_name': 'Game', 'concatenated_reviews': source}
        selected = {'review_id': '1', 'timestamp': '2020-01-01 00:00:00', 'review_summary': 'Good.'}
        output = {**row, 'positive_reviews': [selected, selected, {**selected, 'review_id': 'missing'}],
                  'negative_reviews': []}
        audit = audit_outputs([row], [output], {1: {'1': True}}, limit=2)
        self.assertEqual(audit['counts']['invalid_ids'], 1)
        self.assertEqual(audit['counts']['duplicate_selections'], 1)
        self.assertEqual(audit['vader_agreement_per_requested_slot'], .25)


if __name__ == '__main__':
    unittest.main()
