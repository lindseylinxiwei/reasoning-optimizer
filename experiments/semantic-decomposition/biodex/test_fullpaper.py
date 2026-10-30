import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from fullpaper import run_phase, summarize_stage


class FullPaperTests(unittest.TestCase):
    def test_parallel_documents_have_separate_state_and_resume_without_calls(self):
        barrier = threading.Barrier(2)
        class FakeDecisions:
            def __init__(self, folder):
                self.folder = folder
        def operation(row, decide):
            self.assertEqual(decide.folder.parent.name, row['id'])
            barrier.wait(timeout=5)
            return {'id': row['id']}
        with tempfile.TemporaryDirectory() as temp, patch('iterative_evidence.Decisions', FakeDecisions):
            rows = [{'id':'A'}, {'id':'B'}]
            result, metrics, phase = run_phase(rows, Path(temp), 'scan', 2, operation)
            self.assertEqual(set(result), {'A','B'})
            self.assertEqual(phase['completed_documents'], 2)
            totals = summarize_stage(metrics, phase)
            self.assertEqual(totals['known_api_cost_usd'], 0)
            self.assertEqual(totals['phase_wall_seconds'], phase['process_wall_seconds'])
            again, _, _ = run_phase(rows, Path(temp), 'scan', 2, lambda *_: self.fail('must resume'))
            self.assertEqual(result, again)

    def test_phase_failures_are_recorded_and_not_silently_excluded(self):
        with tempfile.TemporaryDirectory() as temp, patch('iterative_evidence.Decisions', lambda _: None):
            def fail(*_):
                raise ValueError('test failure')
            with self.assertRaises(RuntimeError):
                run_phase([{'id':'A'}], Path(temp), 'scan', 1, fail)
            phase = json.loads((Path(temp)/'scan_phase.json').read_text())
            self.assertEqual(phase['completed_documents'], 0)
            self.assertEqual(phase['errors']['A']['error_type'], 'ValueError')


if __name__ == '__main__':
    unittest.main()
