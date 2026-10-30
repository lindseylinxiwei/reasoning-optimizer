import threading
import unittest
from run_extraction import overlap_stages


class OverlapTests(unittest.TestCase):
    def test_next_extraction_runs_while_selection_waits(self):
        next_extraction = threading.Event()
        caller = threading.get_ident()
        def extract(n):
            self.assertEqual(threading.get_ident(), caller)
            if n == 2:
                next_extraction.set()
            return n
        def finish(n):
            self.assertNotEqual(threading.get_ident(), caller)
            if n == 1:
                self.assertTrue(next_extraction.wait(2))
            return n * 10
        self.assertEqual(list(overlap_stages([1,2,3],extract,finish,True)),[10,20,30])

    def test_sequential_and_error_propagation(self):
        events=[]
        def extract(n): events.append(('extract',n)); return n
        def finish(n): events.append(('finish',n)); return n
        self.assertEqual(list(overlap_stages([1,2],extract,finish)),[1,2])
        self.assertEqual(events,[('extract',1),('finish',1),('extract',2),('finish',2)])
        def failed(n): raise RuntimeError('API failure')
        with self.assertRaises(RuntimeError):
            list(overlap_stages([1,2],extract,failed,True))
