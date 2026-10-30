import copy
import json
import unittest

import numpy as np

from hierarchy import build_hierarchy, validate_hierarchy


class HierarchyTests(unittest.TestCase):
    def test_full_coverage_and_representative_membership(self):
        vectors = np.random.default_rng(12).normal(size=(151, 12))
        tree = build_hierarchy([f"official-{i}" for i in range(151)], vectors,
                               leaf_size=9, branching=4, representatives=5)
        self.assertTrue(validate_hierarchy(json.loads(json.dumps(tree)), 151))
        leaves = [n for n in tree["nodes"].values() if not n["children"]]
        self.assertEqual(sorted(i for n in leaves for i in n["label_indices"]), list(range(151)))
        self.assertLessEqual(max(len(n["label_indices"]) for n in leaves), 9)
        for node in tree["nodes"].values():
            self.assertTrue(set(node["representative_indices"]) <= set(node["label_indices"]))

    def test_identical_vectors_and_zero_vectors_terminate(self):
        for vectors in (np.ones((73, 8)), np.zeros((73, 8))):
            tree = build_hierarchy([str(i) for i in range(73)], vectors,
                                   leaf_size=5, branching=3)
            self.assertTrue(validate_hierarchy(tree, 73))
            self.assertEqual(tree["nodes"][tree["root"]]["split_fallback"], "stable_even_partition")

    def test_deterministic_topology_and_representatives(self):
        vectors = np.random.default_rng(42).normal(size=(97, 15))
        labels = [str(i) for i in range(97)]
        first = build_hierarchy(labels, vectors, leaf_size=7, branching=4, seed=7)
        second = build_hierarchy(labels, vectors, leaf_size=7, branching=4, seed=7)
        self.assertEqual(first, second)

    def test_representatives_include_diverse_extreme(self):
        # The abundant east-facing labels determine the centroid medoid. The
        # rare west-facing label must be selected next by farthest-first.
        vectors = np.array([[1, 0]] * 8 + [[0, 1], [-1, 0]], dtype=float)
        tree = build_hierarchy([str(i) for i in range(10)], vectors,
                               leaf_size=20, representatives=2)
        self.assertEqual(tree["nodes"][tree["root"]]["representative_indices"], [0, 9])

    def test_validator_rejects_invalid_representative_and_coverage(self):
        tree = build_hierarchy([str(i) for i in range(12)], np.eye(12), leaf_size=3)
        broken = copy.deepcopy(tree)
        leaf = next(n for n in broken["nodes"].values() if not n["children"])
        leaf["representative_indices"][0] = 12
        with self.assertRaises(ValueError):
            validate_hierarchy(broken, 12)
        broken = copy.deepcopy(tree)
        broken["nodes"][broken["root"]]["label_indices"].pop()
        with self.assertRaises(ValueError):
            validate_hierarchy(broken, 12)


if __name__ == "__main__":
    unittest.main()
