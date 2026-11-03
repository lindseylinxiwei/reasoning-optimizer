"""Task-independent, source-preserving index for query-time Jev navigation.

The source-structure policy recovers document headings, not task candidates.
Structural heuristics are deliberately identified separately from capacity nodes;
this module does not claim a learned semantic index or retrieval-quality gain.
No models, annotations, task/category strings, or external dependencies are used.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import re
from typing import Any, Mapping, Sequence

try:
    from ._source_structure_snapshot import parse as _parse_numbered, toc_ranges
except ImportError:  # Standalone research scripts.
    from _source_structure_snapshot import parse as _parse_numbered, toc_ranges


FORMAT_VERSION = "source-index-1"
POLICIES = ("source-structure", "balanced-paragraph")


@dataclass(frozen=True)
class IndexNode:
    id: str
    start: int
    end: int
    title: str
    children: tuple[str, ...] = ()
    parent: str | None = None


@dataclass
class SemanticIndex:
    text: str
    root_id: str
    nodes: dict[str, IndexNode]
    policy: str = "source-structure"
    node_kinds: dict[str, str] = field(default_factory=dict)
    links: dict[str, dict[str, tuple[str, ...]]] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def source_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()

    def text_for(self, node_id: str) -> str:
        node = self.nodes[node_id]
        return self.text[node.start:node.end]

    def paragraph_offsets(self) -> tuple[tuple[int, int], ...]:
        return paragraph_offsets(self.text)

    def leaves(self) -> tuple[IndexNode, ...]:
        return tuple(sorted((node for node in self.nodes.values() if not node.children),
                            key=lambda node: (node.start, node.end, node.id)))

    def validate(self) -> None:
        """Reject gaps, overlaps, cycles, unreachable nodes and broken links."""
        if not isinstance(self.text, str) or self.root_id not in self.nodes:
            raise ValueError("An original source and an existing root are required")
        root = self.nodes[self.root_id]
        if (root.start, root.end, root.parent) != (0, len(self.text), None):
            raise ValueError("Root must cover the exact complete source")
        seen: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in seen:
                raise ValueError("Cycle or multiply-parented node")
            seen.add(node_id)
            node = self.nodes[node_id]
            if (not isinstance(node, IndexNode) or node.id != node_id
                    or isinstance(node.start, bool) or isinstance(node.end, bool)
                    or not isinstance(node.start, int) or not isinstance(node.end, int)
                    or not isinstance(node.title, str) or not isinstance(node.children, tuple)
                    or not 0 <= node.start <= node.end <= len(self.text)):
                raise ValueError("Invalid node identity or source interval")
            if node.start == node.end and node_id != self.root_id:
                raise ValueError("Only an empty document root may have zero length")
            cursor = node.start
            for child_id in node.children:
                if child_id not in self.nodes:
                    raise ValueError("Missing child")
                child = self.nodes[child_id]
                if child.parent != node_id or child.start != cursor or child.end > node.end:
                    raise ValueError("Children must exactly partition parent source")
                visit(child_id)
                cursor = child.end
            if node.children and cursor != node.end:
                raise ValueError("Children leave source uncovered")

        visit(self.root_id)
        if seen != set(self.nodes):
            raise ValueError("Unreachable nodes in index")
        if any(node_id not in seen for node_id in self.node_kinds):
            raise ValueError("Kind metadata refers to missing nodes")
        for node_id, relations in self.links.items():
            if node_id not in seen:
                raise ValueError("Link source missing")
            for relation, targets in relations.items():
                if relation not in {"previous", "next", "topic"}:
                    raise ValueError("Unsupported link relation")
                if len(set(targets)) != len(targets) or any(target not in seen or target == node_id for target in targets):
                    raise ValueError("Invalid link target")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "format_version": FORMAT_VERSION, "text": self.text,
            "source_sha256": self.source_hash, "root_id": self.root_id,
            "policy": self.policy,
            "nodes": {key: asdict(self.nodes[key]) for key in sorted(self.nodes)},
            "node_kinds": dict(sorted(self.node_kinds.items())),
            "links": {key: dict(sorted(value.items())) for key, value in sorted(self.links.items())},
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_json(cls, value: str | Mapping[str, Any]) -> "SemanticIndex":
        data = json.loads(value) if isinstance(value, str) else dict(value)
        if data.get("format_version") != FORMAT_VERSION:
            raise ValueError("Unsupported source-index serialization version")
        nodes = {key: IndexNode(**{**node, "children": tuple(node["children"])})
                 for key, node in data["nodes"].items()}
        index = cls(text=data["text"], root_id=data["root_id"], nodes=nodes,
                    policy=data["policy"], node_kinds=data.get("node_kinds", {}),
                    links={key: {relation: tuple(targets) for relation, targets in relations.items()}
                           for key, relations in data.get("links", {}).items()},
                    metadata=data.get("metadata", {}))
        if index.source_hash != data.get("source_sha256"):
            raise ValueError("Serialized source hash does not match original text")
        index.validate()
        return index


def paragraph_offsets(text: str, start: int = 0, end: int | None = None) -> tuple[tuple[int, int], ...]:
    """Partition source at blank lines, retaining every whitespace character.

    Separator whitespace belongs to the preceding paragraph. Empty sources have
    no paragraphs. These are source partitions, not normalized paragraph text.
    """
    if not isinstance(text, str):
        raise TypeError("Original source must be text")
    end = len(text) if end is None else end
    if not 0 <= start <= end <= len(text):
        raise ValueError("Invalid paragraph interval")
    if start == end:
        return ()
    cuts = [start]
    for match in re.finditer(r"\n[ \t\r]*\n(?:[ \t\r]*\n)*", text[start:end]):
        cut = start + match.end()
        if cut < end:
            cuts.append(cut)
    cuts.append(end)
    return tuple(zip(cuts, cuts[1:]))


def balanced_partition(intervals: Sequence[tuple[int, int]], fanout: int = 4) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Contiguous near-equal source-length groups without breaking input units."""
    if fanout < 2:
        raise ValueError("fanout must be at least two")
    if not intervals:
        return ()
    for i, (start, end) in enumerate(intervals):
        if start >= end or (i and intervals[i - 1][1] != start):
            raise ValueError("Partition units must be nonempty and contiguous")
    groups = min(fanout, len(intervals))
    prefix = [0]
    for start, end in intervals:
        prefix.append(prefix[-1] + end - start)
    boundaries = [0]
    for group in range(1, groups):
        lower = boundaries[-1] + 1
        upper = len(intervals) - (groups - group)
        target = prefix[-1] * group / groups
        best = min(range(lower, upper + 1), key=lambda i: (abs(prefix[i] - target), i))
        boundaries.append(best)
    boundaries.append(len(intervals))
    return tuple(tuple(intervals[a:b]) for a, b in zip(boundaries, boundaries[1:]))


def _split_long_paragraph(text: str, start: int, end: int, leaf_chars: int) -> list[tuple[int, int]]:
    """Capacity subdivisions only when a single paragraph exceeds the target."""
    units = []
    while end - start > leaf_chars:
        ceiling = start + leaf_chars
        floor = start + max(1, leaf_chars // 2)
        # Prefer complete source sentences; then a word boundary; last resort is
        # a character boundary. No source text is dropped or normalized.
        sentence_ends = [start + match.end() for match in re.finditer(r"[.!?][\"')\]]*\s+", text[start:ceiling])
                         if start + match.end() >= floor]
        word_ends = [start + match.end() for match in re.finditer(r"\s+", text[start:ceiling])
                     if start + match.end() >= floor]
        cut = sentence_ends[-1] if sentence_ends else (word_ends[-1] if word_ends else ceiling)
        units.append((start, cut))
        start = cut
    if start < end:
        units.append((start, end))
    return units


def _generic_headings(text: str, start: int, end: int, title_end: int,
                      toc: Sequence[tuple[int, int]]) -> list[dict[str, Any]]:
    """Recover explicit markdown/presentation headings in a structural leaf."""
    headings = []
    lines = list(re.finditer(r"[^\n]+", text[start:end]))
    for i, match in enumerate(lines):
        offset = start + match.start()
        stripped = match.group().strip()
        offset += len(match.group()) - len(match.group().lstrip())
        if offset < title_end or not stripped:
            continue
        is_toc_title = bool(re.fullmatch(r"TABLE\s+OF\s+CONTENTS", stripped, re.I))
        if any(a <= offset < b for a, b in toc) and not is_toc_title:
            continue
        markdown = re.fullmatch(r"(#{1,6})\s+(.{1,120}?)(?:\s+#+)?", stripped)
        underline = (i + 1 < len(lines) and bool(re.fullmatch(r"\s*(?:={3,}|-{3,})\s*", lines[i + 1].group())))
        uppercase = (3 <= len(stripped) <= 100 and len(stripped.split()) <= 12
                     and any(c.isalpha() for c in stripped) and stripped.isupper()
                     and not re.search(r"[.;!?]$", stripped))
        if markdown or underline or uppercase:
            headings.append({"start": offset, "title": stripped,
                             "level": len(markdown.group(1)) if markdown else 1,
                             "kind": "markdown-heading" if markdown else ("underlined-heading" if underline else "display-heading"),
                             "children": []})
    # A standalone document title and the first heading may share a level. They
    # remain separate source sections; the parser never discards a title region.
    return headings


def _generic_tree(text: str, start: int, end: int, title_end: int,
                  toc: Sequence[tuple[int, int]]) -> list[dict[str, Any]]:
    roots: list[dict[str, Any]] = []
    stack: list[dict[str, Any]] = []
    for heading in _generic_headings(text, start, end, title_end, toc):
        while stack and stack[-1]["level"] >= heading["level"]:
            stack.pop()
        (stack[-1]["children"] if stack else roots).append(heading)
        stack.append(heading)

    def close(children: list[dict[str, Any]], enclosing_end: int) -> None:
        for i, child in enumerate(children):
            child["end"] = children[i + 1]["start"] if i + 1 < len(children) else enclosing_end
            child["title_end"] = child["start"] + len(child["title"])
            close(child["children"], child["end"])
    close(roots, end)
    return roots


class _Builder:
    def __init__(self, text: str, policy: str, fanout: int, leaf_chars: int):
        self.text, self.policy, self.fanout, self.leaf_chars = text, policy, fanout, leaf_chars
        self.nodes: dict[str, IndexNode] = {}
        self.kinds: dict[str, str] = {}
        self.toc = tuple(toc_ranges(text)) if policy == "source-structure" else ()

    def node(self, start: int, end: int, title: str, kind: str, parent: str | None) -> str:
        stem = f"{kind}:{start}:{end}"
        node_id, suffix = stem, 1
        while node_id in self.nodes:
            node_id = f"{stem}:{suffix}"
            suffix += 1
        self.nodes[node_id] = IndexNode(node_id, start, end, title, (), parent)
        self.kinds[node_id] = kind
        return node_id

    def children(self, node_id: str, children: Sequence[str]) -> None:
        old = self.nodes[node_id]
        self.nodes[node_id] = IndexNode(old.id, old.start, old.end, old.title, tuple(children), old.parent)

    def capacity(self, start: int, end: int, parent: str, title: str = "Source continuation") -> str:
        units = [piece for a, b in paragraph_offsets(self.text, start, end)
                 for piece in _split_long_paragraph(self.text, a, b, self.leaf_chars)]
        return self.capacity_units(units, parent, title)

    def capacity_units(self, units: Sequence[tuple[int, int]], parent: str, title: str) -> str:
        start, end = units[0][0], units[-1][1]
        leaf = end - start <= self.leaf_chars
        node_id = self.node(start, end, title, "source-window" if leaf else "capacity-group", parent)
        if not leaf:
            groups = balanced_partition(units, self.fanout)
            # Long paragraphs have already been subdivided, so a one-unit large
            # group cannot occur; this guard rejects regressions rather than loop.
            if len(groups) < 2:
                raise ValueError("Unpartitionable capacity span")
            child_ids = [self.capacity_units(group, node_id, "Source continuation") for group in groups]
            self.children(node_id, child_ids)
        return node_id

    def structure(self, raw: Mapping[str, Any], parent: str | None) -> str:
        start, end = raw["start"], raw["end"]
        kind = raw.get("kind", "source-section")
        title = "Document" if parent is None else raw["title"]
        node_id = self.node(start, end, title, kind, parent)
        descendants = list(raw.get("children", []))
        if not descendants:
            descendants = _generic_tree(self.text, start, end, raw.get("title_end", start), self.toc)
        children = []
        cursor = start
        for child in descendants:
            if child["start"] > cursor:
                children.append(self.capacity(cursor, child["start"], node_id,
                                              "Section introduction" if cursor == start else "Source continuation"))
            children.append(self.structure(child, node_id))
            cursor = child["end"]
        if children and cursor < end:
            children.append(self.capacity(cursor, end, node_id))
        elif not children and end - start > self.leaf_chars:
            children.append(self.capacity(start, end, node_id))
        self.children(node_id, children)
        return node_id


def _add_links(index: SemanticIndex) -> None:
    leaves = index.leaves()
    for i, node in enumerate(leaves):
        index.links[node.id] = {
            "previous": (leaves[i - 1].id,) if i else (),
            "next": (leaves[i + 1].id,) if i + 1 < len(leaves) else (),
            "topic": (),
        }
    semantic = [node for node in index.nodes.values()
                if index.node_kinds.get(node.id) in {"article", "clause", "markdown-heading", "underlined-heading", "display-heading"}]
    # Heading lexical overlap is a transparent weak topic link, not an embedding
    # or relevance judgment. Numeric labels and common structural words are out.
    ignored = {"article", "section", "chapter", "the", "and", "of", "for", "to", "a", "an"}
    terms = {node.id: set(re.findall(r"[a-z]{3,}", node.title.lower())) - ignored for node in semantic}
    for node in semantic:
        candidates = []
        for other in semantic:
            if node.id == other.id or max(node.start, other.start) < min(node.end, other.end):
                continue  # No links to an overlapping ancestor/descendant.
            common = terms[node.id] & terms[other.id]
            if common:
                candidates.append((-len(common), abs(node.start - other.start), other.id))
        relation = index.links.setdefault(node.id, {"previous": (), "next": (), "topic": ()})
        relation["topic"] = tuple(value[2] for value in sorted(candidates)[:4])


def build_index(text: str, policy: str = "source-structure", fanout: int = 4,
                leaf_chars: int = 2400) -> SemanticIndex:
    """Build once from original text, independently of extraction queries.

    Source headings are never merged merely to meet fanout; large semantic nodes
    are legal. Only capacity groups obey fanout. Query-time request paging is the
    navigator's responsibility. Empty documents have a valid empty root.
    """
    if not isinstance(text, str):
        raise TypeError("Original source must be text")
    if policy not in POLICIES:
        raise ValueError(f"Unknown index policy: {policy}")
    if isinstance(fanout, bool) or not isinstance(fanout, int) or fanout < 2:
        raise ValueError("fanout must be an integer of at least two")
    if isinstance(leaf_chars, bool) or not isinstance(leaf_chars, int) or leaf_chars < 32:
        raise ValueError("leaf_chars must be an integer of at least 32")
    builder = _Builder(text, policy, fanout, leaf_chars)
    if not text:
        root_id = builder.node(0, 0, "Document", "root", None)
        parser_metadata = {}
    elif policy == "source-structure":
        parsed = _parse_numbered(text)
        root_id = builder.structure(parsed["semantic"], None)
        parser_metadata = {"numbered_headings": len(parsed["candidates"]),
                           "excluded_numbered_headings": parsed["excluded_candidates"],
                           "table_of_contents_ranges": parsed["toc_ranges"]}
    else:
        root_id = builder.node(0, len(text), "Document", "root", None)
        if len(text) > leaf_chars:
            builder.children(root_id, [builder.capacity(0, len(text), root_id)])
        parser_metadata = {}
    index = SemanticIndex(text, root_id, builder.nodes, policy, builder.kinds,
                          metadata={"query_independent": True, "source_preserved": True,
                                    "classification": ("deterministic source structure; heading heuristics are unverified"
                                                       if policy == "source-structure" else "balanced source-capacity control; not a semantic tree"),
                                    "fanout": fanout, "leaf_chars": leaf_chars,
                                    "long_paragraph_policy": "sentence/word/character subdivision only above leaf target",
                                    "topic_link_policy": "up to four nonoverlapping headings with lexical title overlap",
                                    **parser_metadata})
    _add_links(index)
    index.validate()
    return index
