"""Exercise preparation and all three modes without network access."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import run


class RunnerTest(unittest.TestCase):
    def test_three_methods_share_inputs_and_keep_gt_out(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'source.json'
            source.write_text(json.dumps([{'id': '1', 'fulltext_processed': 'nausea fever',
                'possible_labels': 'nausea\nfever\nrash', 'ground_truth_reactions': ['nausea']}]))
            class FakeEmbeddings:
                cache_hits = 0
                new_vectors = 0
                def __init__(self, *args): pass
                def __call__(self, texts):
                    return np.asarray([[1, 0] if t != 'rash' else [0, 1] for t in texts])
                def close(self): pass
            with patch.object(run, 'Embeddings', FakeEmbeddings):
                run.prepare(argparse.Namespace(input=source, output=root/'prepared', cache=root/'cache',
                    max_candidates=3, passage_tokens=256, overlap_tokens=64))
            prepared = json.loads((root/'prepared/prepared.json').read_text())
            self.assertNotIn('ground_truth', json.dumps(prepared))
            seen = []
            class FakeRanker:
                def __init__(self, ledger): pass
                def __call__(self, evidence, candidates, limit):
                    seen.append((evidence, candidates, limit))
                    return [candidates[0]['id']], {}
            with patch.object(run, 'JevRanker', FakeRanker), patch.object(run, 'GeminiRanker', FakeRanker):
                run.run(argparse.Namespace(prepared=root/'prepared', output=root/'results',
                    k=2, output_limit=2, method='all'))
            self.assertEqual(seen[0], seen[1])
            summary = json.loads((root/'results/summary.json').read_text())
            self.assertEqual(summary['candidate_recall'], 1)
            self.assertEqual(summary['oracle_rp5'], 1)
            for method in ['embedding', 'jev', 'gemini']:
                self.assertEqual(summary['methods'][method]['evaluation']['avg_rp_at_5'], 1)
                result = json.loads((root/'results'/method/'results.json').read_text())[0]
                self.assertEqual(result['candidate_ids'], ['L00000', 'L00001'])
            source.write_text(source.read_text() + ' ')
            with self.assertRaisesRegex(ValueError, 'Source changed'):
                run.run(argparse.Namespace(prepared=root/'prepared', output=root/'changed',
                    k=2, output_limit=2, method='embedding'))


if __name__ == '__main__':
    unittest.main()
