"""Boundary/abstention tests; no model downloads or API calls."""
import re
import unittest
from pipeline import extract_candidates, select_and_copy, windows


def extractor(text):
    return [{"start": m.start(), "end": m.end(), "text": m.group(), "score": 0.9}
            for m in re.finditer(r"Apple|Microsoft", text)]


class PipelineTests(unittest.TestCase):
    def test_full_coverage_unicode_and_overlap_deduplication(self):
        text = "é 😀 Apple Microsoft filler Apple trailing"
        result = extract_candidates(text, extractor, chunk_size=4, overlap=2)
        self.assertGreater(result["window_count"], 1)
        self.assertEqual(result["mention_count"], 3)
        self.assertEqual([c["text"] for c in result["candidates"]], ["Apple", "Microsoft"])
        self.assertEqual(len(result["candidates"][0]["mentions"]), 2)
        for candidate in result["candidates"]:
            for span in candidate["mentions"]:
                self.assertEqual(text[span["start"]:span["end"]], candidate["text"])
        slices=list(windows(text, 4, 2))
        self.assertEqual(slices[-1][0]+len(slices[-1][1]),len(text))

    def test_batch_windows_preserve_offsets_and_overlap_deduplication(self):
        class Batched:
            def extract_batch(self, texts, batch_size):
                return [extractor(text) for text in texts]
        text = "é 😀 Apple Microsoft filler Apple trailing"
        serial = extract_candidates(text, extractor, 4, 2)
        batch = extract_candidates(text, Batched(), 4, 2, batch_size=8)
        self.assertEqual(serial, batch)
        class Broken:
            def extract_batch(self, texts, batch_size):
                return []
        with self.assertRaises(ValueError):
            extract_candidates(text, Broken(), batch_size=8)

    def test_copy_cannot_use_generated_name(self):
        text = "Apple and Microsoft"
        candidates = extract_candidates(text, extractor)["candidates"]
        result = select_and_copy(text, candidates, lambda *args: {
            "candidate_id": "C002", "company_name": "Fabricated name", "cost_usd": 0.1})
        self.assertEqual(result["company_name"], "Microsoft")

    def test_none_and_empty_candidates(self):
        def forbidden(*args):
            self.fail("No selector call should occur without candidates")
        self.assertEqual(select_and_copy("text", [], forbidden)["reason"], "no_candidates")
        candidates = extract_candidates("Apple", extractor)["candidates"]
        result = select_and_copy("Apple", candidates, lambda *args: {"candidate_id": "NONE"})
        self.assertEqual(result["reason"], "selector_none")
        self.assertIsNone(result["company_name"])

    def test_invalid_decision_and_transport_errors_are_not_abstentions(self):
        candidates = extract_candidates("Apple", extractor)["candidates"]
        with self.assertRaises(ValueError):
            select_and_copy("Apple", candidates, lambda *args: {"candidate_id": "invented"})
        def failed(*args):
            raise TimeoutError("Unavailable")
        with self.assertRaises(TimeoutError):
            select_and_copy("Apple", candidates, failed)

    def test_invalid_spans_are_rejected(self):
        result = extract_candidates("Apple", lambda text: [{"start": 0, "end": 4, "text": "Apple"}])
        self.assertEqual(result["candidates"], [])
        self.assertEqual(len(result["invalid_spans"]), 1)

    def test_window_boundaries_and_empty_input(self):
        self.assertEqual(list(windows("  ")), [])
        self.assertEqual(len(list(windows("a b c d", 4, 2))), 1)
        with self.assertRaises(ValueError):
            list(windows("text", 4, 4))


if __name__ == "__main__":
    unittest.main()
