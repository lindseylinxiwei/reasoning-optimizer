"""Offline structural integrity checks; no contract gold or provider calls."""
from dataclasses import replace
import hashlib
import json
import random
import unittest

from semantic_index import IndexNode, SemanticIndex, balanced_partition, build_index, paragraph_offsets


class SourceIndexTests(unittest.TestCase):
    def assert_source_preserved(self, index):
        index.validate()
        self.assertEqual("".join(index.text_for(node.id) for node in index.leaves()), index.text)
        cursor = 0
        for leaf in index.leaves():
            self.assertEqual(leaf.start, cursor)
            cursor = leaf.end
        self.assertEqual(cursor, len(index.text))
        self.assertEqual(index.source_hash, hashlib.sha256(index.text.encode("utf-8")).hexdigest())

    def test_flattened_numbered_heading_and_article_nesting(self):
        text = ("Preamble kept.\n\nARTICLE ELEVEN  GENERAL TERMS  11.1 Transfer. The holder may transfer "
                "this instrument.  11.2 Applicable Rules. The following rules apply.\n\n"
                "ARTICLE TWELVE\nREPORTING\n12.1 Reports. Monthly reports are required.")
        index = build_index(text, leaf_chars=64)
        self.assert_source_preserved(index)
        articles = [node for node in index.nodes.values() if index.node_kinds[node.id] == "article"]
        self.assertEqual(len(articles), 2)
        first = min(articles, key=lambda node: node.start)
        clauses = [index.nodes[child] for child in first.children if index.node_kinds[child] == "clause"]
        self.assertEqual([node.title for node in clauses], ["11.1 Transfer.", "11.2 Applicable Rules."])
        self.assertTrue(any(index.node_kinds[child] == "source-window" for child in first.children))
        self.assertGreater(first.end - first.start, 64)  # Semantic node stays whole.

    def test_real_prefix_nesting_and_repeated_number_reset(self):
        text = ("1 First. Text.\n1.1 Child. More.\n1.1.1 Detail. Detail text.\n"
                "2 Second. Other.\n2.1 Child. More.\n1 Restart. Final text.")
        index = build_index(text)
        self.assert_source_preserved(index)
        by_title = {node.title: node for node in index.nodes.values()}
        self.assertEqual(by_title["1.1.1 Detail."].parent, by_title["1.1 Child."].id)
        self.assertEqual(by_title["2.1 Child."].parent, by_title["2 Second."].id)
        self.assertEqual(by_title["1 Restart."].parent, index.root_id)

    def test_toc_excluded_from_numbered_headings_but_kept_reachable(self):
        text = ("TABLE OF CONTENTS\n\nARTICLE I PURPOSE .... 1\n1.1 Introduction .... 2\n\n"
                "This document is provided for readers and the terms that follow are the operative material "
                "and are intended to apply to everyone.\n\nARTICLE I PURPOSE\n1.1 Introduction. The text follows.")
        index = build_index(text, leaf_chars=96)
        self.assert_source_preserved(index)
        clauses = [node for node in index.nodes.values() if index.node_kinds[node.id] == "clause"]
        self.assertEqual(len(clauses), 1)
        self.assertTrue(index.metadata["excluded_numbered_headings"])
        self.assertIn("ARTICLE I PURPOSE .... 1", "".join(index.text_for(node.id) for node in index.leaves()))

    def test_markdown_structure_for_nonlegal_text(self):
        text = ("A field guide.\n\n# Biology\nAn overview.\n\n## Cell structure\nCells contain membranes.\n\n"
                "## Energy\nEnergy is transferred.\n\n# Chemistry\nAtoms form compounds.")
        index = build_index(text)
        self.assert_source_preserved(index)
        by_title = {node.title: node for node in index.nodes.values()}
        self.assertEqual(by_title["## Cell structure"].parent, by_title["# Biology"].id)
        self.assertEqual(by_title["## Energy"].parent, by_title["# Biology"].id)
        self.assertEqual(by_title["# Chemistry"].parent, index.root_id)

    def test_display_and_underlined_headings_are_generic(self):
        text = "INTRODUCTION\nA general description.\n\nMethods\n-------\nA method description.\n\nRESULTS\nSome results."
        index = build_index(text)
        self.assert_source_preserved(index)
        self.assertTrue({"INTRODUCTION", "Methods", "RESULTS"}.issubset({node.title for node in index.nodes.values()}))

    def test_structure_policy_and_capacity_control_are_distinct(self):
        text = "# Travel\n" + ("A trip occurred. " * 30) + "\n\n# Food\n" + ("A meal was served. " * 30)
        source = build_index(text, leaf_chars=64)
        control = build_index(text, policy="balanced-paragraph", leaf_chars=64)
        self.assert_source_preserved(source)
        self.assert_source_preserved(control)
        self.assertIn("not a semantic tree", control.metadata["classification"])
        self.assertNotEqual(source.to_json(), control.to_json())
        self.assertFalse(any("heading" in kind for kind in control.node_kinds.values()))

    def test_paragraph_boundaries_and_whitespace_preserved(self):
        text = "  First.\n\n\n Second.\r\n\r\nThird.  "
        intervals = paragraph_offsets(text)
        self.assertEqual(len(intervals), 3)
        self.assertEqual("".join(text[a:b] for a, b in intervals), text)
        self.assertEqual(intervals[0][0], 0)
        self.assertEqual(intervals[-1][1], len(text))

    def test_character_balancing_does_not_split_supplied_units(self):
        lengths = [100, 900, 100, 100, 100, 100, 100, 100]
        units = []
        offset = 0
        for length in lengths:
            units.append((offset, offset + length))
            offset += length
        groups = balanced_partition(units, 3)
        self.assertEqual(tuple(piece for group in groups for piece in group), tuple(units))
        self.assertEqual(len(groups), 3)
        self.assertEqual(sum(b - a for a, b in groups[0]), 100)  # Avoid forcing long unit into first group.

    def test_capacity_nodes_bounded_and_long_paragraphs_reachable(self):
        text = "x" * 211 + "\n\n" + "word " * 117 + "\n\n" + "Short paragraph."
        for policy in ("source-structure", "balanced-paragraph"):
            index = build_index(text, policy=policy, fanout=3, leaf_chars=64)
            self.assert_source_preserved(index)
            for node in index.nodes.values():
                if index.node_kinds[node.id] == "capacity-group":
                    self.assertLessEqual(len(node.children), 3)
            self.assertTrue(all(node.end - node.start <= 64 for node in index.leaves()))

    def test_roundtrip_is_deterministic_and_rejects_source_mutation(self):
        text = "# 中文 α\nOriginal Unicode text.\n\n# Other\nA second section."
        first = build_index(text, leaf_chars=64)
        second = build_index(text, leaf_chars=64)
        self.assertEqual(first.to_json(), second.to_json())
        self.assertEqual(first.to_json(), SemanticIndex.from_json(first.to_json()).to_json())
        mutated = json.loads(first.to_json())
        mutated["text"] += "!"
        with self.assertRaisesRegex(ValueError, "hash"):
            SemanticIndex.from_json(mutated)

    def test_neighbors_and_title_topic_links(self):
        text = "# Energy systems\nOne account.\n\n# Other topic\nAnother.\n\n# Energy storage\nA third account."
        index = build_index(text, leaf_chars=64)
        headings = {node.title: node for node in index.nodes.values() if index.node_kinds[node.id] == "markdown-heading"}
        self.assertIn(headings["# Energy storage"].id, index.links[headings["# Energy systems"].id]["topic"])
        leaves = index.leaves()
        self.assertEqual(index.links[leaves[0].id]["next"], (leaves[1].id,))
        self.assertEqual(index.links[leaves[1].id]["previous"], (leaves[0].id,))

    def test_validation_rejects_gap_overlap_unreachable_and_wrong_parent(self):
        root = IndexNode("root", 0, 4, "Document", ("a", "b"), None)
        a = IndexNode("a", 0, 2, "A", (), "root")
        b = IndexNode("b", 2, 4, "B", (), "root")
        SemanticIndex("abcd", "root", {"root": root, "a": a, "b": b}).validate()
        for broken in (replace(b, start=3), replace(b, start=1), replace(b, parent=None)):
            with self.assertRaises(ValueError):
                SemanticIndex("abcd", "root", {"root": root, "a": a, "b": broken}).validate()
        with self.assertRaisesRegex(ValueError, "Unreachable"):
            SemanticIndex("abcd", "root", {"root": root, "a": a, "b": b,
                                            "c": IndexNode("c", 0, 1, "C")}).validate()

    def test_empty_document_and_parameter_rejection(self):
        empty = build_index("")
        self.assert_source_preserved(empty)
        self.assertEqual(empty.paragraph_offsets(), ())
        self.assertEqual(empty.to_json(), SemanticIndex.from_json(empty.to_json()).to_json())
        for kwargs in ({"policy": "answer-candidates"}, {"fanout": 1}, {"leaf_chars": 0}):
            with self.assertRaises(ValueError):
                build_index("Some text", **kwargs)

    def test_random_unstructured_sources_remain_exact(self):
        rng = random.Random(23917)
        for _ in range(20):
            text = "\n\n".join("".join(rng.choice("ab β你. \t") for _ in range(rng.randint(1, 150)))
                               for _ in range(rng.randint(1, 15)))
            for policy in ("source-structure", "balanced-paragraph"):
                index = build_index(text, policy=policy, fanout=3, leaf_chars=64)
                self.assert_source_preserved(index)
                self.assertEqual(index.to_json(), SemanticIndex.from_json(index.to_json()).to_json())


if __name__ == "__main__":
    unittest.main()
