import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import tiktoken
import core


class RetrievalTests(unittest.TestCase):
    def test_max_passage_similarity_and_stable_ties(self):
        labels = ['A', 'B', 'C']
        result = core.retrieve(labels, [[1, 0], [0, 1], [1, 0]], [{}, {}], [[1, 0], [0, 1]], 3)
        self.assertEqual([r['label'] for r in result], labels)
        self.assertEqual([r['passage_index'] for r in result], [0, 1, 0])
        self.assertEqual(result[:1], core.retrieve(labels, [[1, 0], [0, 1], [1, 0]], [{}, {}], [[1, 0], [0, 1]], 1))

    def test_unicode_passages_are_source_spans(self):
        text = '患者の症状: nausea 🩺 and fever. ' * 40
        encoder = tiktoken.get_encoding('cl100k_base')
        chunks = core.passages(text, encoder, 32, 8)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertEqual(text[chunk['start']:chunk['end']], chunk['text'])
        self.assertEqual(chunks[0]['start'], 0)
        self.assertEqual(chunks[-1]['end'], len(text))
        self.assertTrue(all(a['end'] >= b['start'] for a, b in zip(chunks, chunks[1:])))

    def test_evidence_budget(self):
        encoder = tiktoken.get_encoding('cl100k_base')
        text = 'fever and nausea. ' * 100
        chunks = core.passages(text, encoder, 32, 8)
        candidates = [{'passage_index': 2}]
        evidence, mode = core.shared_evidence(text, chunks, candidates, encoder, 50)
        self.assertEqual(mode, 'retrieved_passages')
        self.assertLessEqual(len(encoder.encode(evidence)), 50)
        self.assertIn(chunks[2]['text'], evidence)
        self.assertEqual(core.shared_evidence('short', chunks, candidates, encoder), ('short', 'full_document'))

    def test_probability_not_winner_confidence(self):
        self.assertEqual(core.support_probability({'confidence': .9, 'probabilities': {'not_supported': .9}}), .09999999999999998)
        for answer in [{'confidence': .9}, {'probabilities': {'supported': float('nan')}}, {'probabilities': {'supported': 2}}]:
            with self.assertRaises(ValueError): core.support_probability(answer)

    def test_reject_duplicate_or_hallucinated_ids(self):
        candidates = [{'id': 'L1'}]
        for ids in [['L1', 'L1'], ['unknown'], 'L1']:
            with self.assertRaises(ValueError): core.validate_ids(ids, candidates, 10)
        self.assertEqual(core.validate_ids([], candidates, 10), [])

    def test_embedding_cache_model_and_content_keys(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            fake = {'data': [{'index': 0, 'embedding': [1, 0]}]}
            with patch.dict(os.environ, {'OPENAI_API_KEY': 'fake'}), patch.object(core, 'post', return_value=fake) as post:
                client = core.Embeddings(path / 'cache.db', path / 'ledger')
                self.assertEqual(client(['a', 'a']).shape, (2, 2))
                client(['a']); self.assertEqual(post.call_count, 1)
                client(['b']); self.assertEqual(post.call_count, 2)
                client.close()
                client = core.Embeddings(path / 'cache.db', path / 'ledger', model='different')
                client(['a']); self.assertEqual(post.call_count, 3)
                client.close()

    def test_post_records_unknown_failed_charge(self):
        with tempfile.TemporaryDirectory() as tmp:
            ledger = Path(tmp) / 'calls.jsonl'
            with patch.object(core.requests, 'post', side_effect=RuntimeError('private detail')):
                with self.assertRaises(RuntimeError): core.post('url', {}, {}, ledger, lambda b: None)
            row = json.loads(ledger.read_text())
            self.assertIsNone(row['cost_usd'])
            self.assertNotIn('private detail', ledger.read_text())

    def test_jev_adapter_keeps_multiple_supported_labels(self):
        from types import SimpleNamespace
        ranker = core.JevRanker.__new__(core.JevRanker)
        answers = {'A': {'choice': 'supported', 'probabilities': {'supported': .6}},
                   'B': {'choice': 'not_supported', 'probabilities': {'supported': .2}},
                   'C': {'choice': 'supported', 'probabilities': {'supported': .9}}}
        ranker.client = SimpleNamespace(_pack=lambda s,q: [q],
            _request=lambda *args: {'answers': answers})
        ids, _ = ranker('paper', [{'id': k, 'label': k} for k in answers], 10)
        self.assertEqual(ids, ['C', 'A'])


if __name__ == '__main__':
    unittest.main()
