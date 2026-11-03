"""Synthetic-only integrity and vector-boundary checks for the topic index."""
from dataclasses import dataclass
import json
import unittest

from semantic_index import SemanticIndex, build_index
from semantic_topic_index import build_topic_index


@dataclass(frozen=True)
class SpanFixture:
    start: int
    end: int


class TopicIndexTests(unittest.TestCase):
    def assert_source_preserved(self, index):
        index.validate()
        self.assertEqual("".join(index.text_for(leaf.id) for leaf in index.leaves()), index.text)
        self.assertEqual(index.to_json(), SemanticIndex.from_json(index.to_json()).to_json())
        self.assertTrue(all(len(node.children) <= index.metadata["fanout"] for node in index.nodes.values()))

    def test_low_similarity_boundary_preferred_within_balance_limits(self):
        text = "a" * 80 + "b" * 160
        windows = [(i, i + 40) for i in range(0, len(text), 40)]
        vectors = [[1, 0]] * 2 + [[0, 1]] * 4
        index = build_topic_index(text, windows, vectors, fanout=2, leaf_chars=40)
        self.assert_source_preserved(index)
        root_children = [index.nodes[child] for child in index.nodes[index.root_id].children]
        self.assertEqual([(node.start, node.end) for node in root_children], [(0, 80), (80, 240)])
        self.assertFalse(index.metadata["partitions"][0]["balance_relaxed"])
        self.assertEqual(index.metadata["partitions"][0]["semantic_cut_cost"], 0.0)

    def test_extreme_semantic_cut_rejected_by_character_balance(self):
        text = "x" * 240
        windows = [(i, i + 40) for i in range(0, 240, 40)]
        vectors = [[1, 0], [-1, 0], [-1, 0], [0, 1], [0, 1], [0, 1]]
        index = build_topic_index(text, windows, vectors, fanout=2, leaf_chars=40)
        children = [index.nodes[child] for child in index.nodes[index.root_id].children]
        self.assertEqual(children[0].end, 120)  # Cut at 40 has cosine -1 but is too imbalanced.
        self.assert_source_preserved(index)

    def test_same_vectors_use_length_then_source_order_tiebreak(self):
        text = "x" * 240
        windows = [(i, i + 40) for i in range(0, 240, 40)]
        index = build_topic_index(text, windows, [[1, 0]] * 6, fanout=2, leaf_chars=40)
        root = index.nodes[index.root_id]
        self.assertEqual(index.nodes[root.children[0]].end, 120)
        second = build_topic_index(text, windows, [[100, 0]] * 6, fanout=2, leaf_chars=40)
        self.assertEqual(index.to_json(), second.to_json())

    def test_overlapping_windows_project_without_source_loss(self):
        text = "Original source " * 10
        end = len(text)
        windows = [SpanFixture(0, 100), SpanFixture(60, end)]
        index = build_topic_index(text, windows, [[1, 0], [0, 1]], fanout=2, leaf_chars=40)
        self.assert_source_preserved(index)
        self.assertEqual(index.metadata["projection_provenance"]["atomic_regions"], 3)
        self.assertTrue(index.metadata["projection_provenance"]["overlapping_windows"])
        self.assertEqual([(leaf.start, leaf.end) for leaf in index.leaves()], [(0, 60), (60, 100), (100, end)])

    def test_opposing_overlapping_vectors_have_counted_stable_fallback(self):
        text = "x" * 120
        index = build_topic_index(text, [(0, 80), (40, 120)], [[1, 0], [-1, 0]], fanout=2, leaf_chars=40)
        self.assert_source_preserved(index)
        self.assertEqual(index.metadata["projection_provenance"]["zero_mean_fallbacks"], 1)

    def test_nonlegal_text_and_control_leaf_units(self):
        text = "# Forests\n" + "Trees retain water. " * 8 + "\n\n# Oceans\n" + "Water supports life. " * 8
        control = build_index(text, policy="balanced-paragraph", leaf_chars=64)
        windows = control.leaves()
        vectors = [[1.0, 0.0] if node.start < len(text) / 2 else [0.0, 1.0] for node in windows]
        index = build_topic_index(text, windows, vectors, leaf_chars=64)
        self.assert_source_preserved(index)
        self.assertEqual(index.policy, "embedding-topic")
        self.assertIn("heuristic", index.metadata["classification"])
        self.assertIn("full standalone", index.metadata["index_cost_policy"])

    def test_source_window_gap_is_rejected_instead_of_filtered(self):
        for windows in ([(0, 40), (50, 120)], [(5, 120)], [(0, 115)]):
            with self.assertRaisesRegex(ValueError, "complete document"):
                build_topic_index("x" * 120, windows, [[1, 0]] * len(windows))

    def test_invalid_vectors_windows_and_configuration_are_rejected(self):
        for vectors in ([[0, 0]], [[float("nan"), 0]], [[True, 0]], [[]]):
            with self.assertRaises(ValueError):
                build_topic_index("x" * 120, [(0, 120)], vectors)
        with self.assertRaisesRegex(ValueError, "dimensions"):
            build_topic_index("x" * 120, [(0, 60), (60, 120)], [[1], [1, 0]])
        with self.assertRaisesRegex(ValueError, "one injected vector"):
            build_topic_index("x" * 120, [(0, 60), (60, 120)], [[1]])
        for kwargs in ({"fanout": 1}, {"leaf_chars": 0}, {"balance_tolerance": 1}, {"balance_tolerance": float("nan")}):
            with self.assertRaises(ValueError):
                build_topic_index("x" * 120, [(0, 120)], [[1]], **kwargs)

    def test_indivisible_large_region_remains_available(self):
        text = "x" * 120
        index = build_topic_index(text, [(0, 120)], [[1, 0]], leaf_chars=40)
        self.assert_source_preserved(index)
        self.assertEqual(len(index.nodes), 1)
        self.assertEqual(index.text_for(index.root_id), text)

    def test_infeasible_balance_uses_explicit_capacity_fallback(self):
        text = "x" * 120
        index = build_topic_index(text, [(0, 10), (10, 20), (20, 120)], [[1, 0], [0, 1], [1, 0]],
                                  fanout=3, leaf_chars=40, balance_tolerance=0.2)
        self.assert_source_preserved(index)
        self.assertTrue(index.metadata["partitions"][0]["balance_relaxed"])
        self.assertIsNone(index.metadata["partitions"][0]["semantic_cut_cost"])

    def test_empty_source_serializes_without_embeddings(self):
        index = build_topic_index("", [], [])
        self.assert_source_preserved(index)
        self.assertEqual(index.metadata["projection_provenance"]["input_windows"], 0)

    def test_json_records_vector_provenance_and_determinism(self):
        text = "海洋 α " * 30
        windows = [{"start": 0, "end": 90}, {"start": 90, "end": len(text)}]
        args = dict(fanout=2, leaf_chars=64)
        first = build_topic_index(text, windows, [[1, 1], [0, 1]], **args)
        second = build_topic_index(text, windows, [[1, 1], [0, 1]], **args)
        self.assertEqual(first.to_json(), second.to_json())
        metadata = json.loads(first.to_json())["metadata"]
        self.assertEqual(len(metadata["projection_provenance"]["normalized_window_vectors_sha256"]), 64)
        self.assert_source_preserved(first)


if __name__ == "__main__":
    unittest.main()
