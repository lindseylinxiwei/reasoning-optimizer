"""Declarative, deterministic index construction for Jev retrieval.

An IndexSpec is chosen once per corpus (e.g. by the optimizer agent after inspecting sample
documents) and applied identically to every document. It is data, not code: an ordered list
of splitters from outer to inner level, plus sizes. Building an index is pure string
processing, so it is cheap and needs no model calls.

Splitters:
  heading     split before every line matching a named `style` (see HEADING_STYLES), or a
              custom `pattern` (a regular expression tested against the start of each line)
  page_break  split at form feeds, or before every line matching `style`/`pattern` if given
  blank_line  split into paragraphs at blank lines
  fixed       split into pieces of about `chars` characters at whitespace

After the last level, regions longer than leaf_chars are cut into leaves at sentence or
whitespace boundaries. Nodes with more than `fanout` children get contiguous groups. The
result is a SemanticIndex that the vendored navigator validates (every character covered
exactly once).
"""
import re
from statistics import median
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from docetl.retrieval.jev.semantic_index import IndexNode, SemanticIndex, balanced_partition

SENTENCE_END = re.compile(r"(?<=[.!?;:])\s+|\n+")
BLANK_LINE = re.compile(r"\n[ \t]*\n\s*")


# Named line styles. Patterns are matched against the start of each line (re.match).
HEADING_STYLES = {
    "numbered": (r"\s*\d{1,2}(\.\d{1,2}){0,3}[.)]?\s+[A-Z][A-Za-z]",
                 "numbered section headings at any depth, e.g. '4 Operations', '4.2 Products'"),
    "numbered_top": (r"\s*\d{1,2}[.)]?\s+[A-Z][A-Za-z]",
                     "top-level numbered headings only, e.g. '4 Operations', '4. Operations'"),
    "numbered_sub": (r"\s*\d{1,2}\.\d{1,2}(\.\d{1,2})?[.)]?\s+[A-Z][A-Za-z]",
                     "sub-section headings only, e.g. '4.2 Products', '4.2.1 Pumps'"),
    "article_section": (r"\s*(ARTICLE|Article|SECTION|Section|CHAPTER|Chapter|PART|Part)\s+[0-9IVXLC]+\b",
                        "legal/book structure, e.g. 'ARTICLE 5', 'Section 12', 'Chapter II'"),
    "markdown": (r"#{1,6}\s+\S", "markdown headings, e.g. '## Our business'"),
    "all_caps": (r"[A-Z][A-Z0-9&'/,-]+(?: [A-Z0-9&'/,-]{2,}){1,7}\s*$",
                 "short ALL CAPS lines of 2-8 words, e.g. 'OUR BUSINESS MODEL'"),
}
PAGE_STYLES = {
    "page_number_line": (r"\s*(Page\s+)?\d{1,4}\s*(/\s*\d{1,4})?\s*$", "lines that hold only a page number"),
}

# Limits a heading level must satisfy on the sample documents (median over documents).
MAX_HEADING_LINE_SHARE = 0.05     # at most 5% of non-empty lines match
MIN_SECTION_CHARS = 1000          # sections average at least this many characters
MIN_PAGE_CHARS = 500              # page-break splits average at least this many characters


class Splitter(BaseModel):
    kind: Literal["heading", "page_break", "blank_line", "fixed"]
    style: Optional[str] = Field(None, description="heading: a HEADING_STYLES name; page_break: a PAGE_STYLES name")
    pattern: Optional[str] = Field(
        None, description="custom regex matched at the start of a line, instead of a style")
    chars: Optional[int] = Field(None, description="fixed: target piece size in characters")

    @property
    def regex(self) -> Optional[str]:
        if self.pattern:
            return self.pattern
        if self.style:
            return (HEADING_STYLES if self.kind == "heading" else PAGE_STYLES)[self.style][0]
        return None

    @model_validator(mode="after")
    def check(self):
        if self.style and self.pattern:
            raise ValueError("Give either a style or a pattern, not both")
        if self.style:
            styles = HEADING_STYLES if self.kind == "heading" else PAGE_STYLES if self.kind == "page_break" else {}
            if self.style not in styles:
                raise ValueError(f"Unknown {self.kind} style {self.style!r}; choose one of {sorted(styles)}")
        if self.kind == "heading" and not self.regex:
            raise ValueError("A heading splitter needs a style or a pattern")
        if self.kind == "fixed" and not (self.chars and self.chars >= 200):
            raise ValueError("A fixed splitter needs chars >= 200")
        if self.pattern:
            try:
                re.compile(self.pattern)
            except re.error as err:
                raise ValueError(f"Invalid pattern {self.pattern!r}: {err}")
        return self


class IndexSpec(BaseModel):
    levels: List[Splitter] = Field(default_factory=list, description="Splitters from outer to inner level (0-4)")
    leaf_chars: int = Field(2400, description="Regions longer than this are cut into leaves")
    min_chars: int = Field(200, description="Segments shorter than this are merged into a neighbor")
    fanout: int = Field(4, description="Maximum children per node before contiguous grouping")

    @field_validator("levels")
    @classmethod
    def check_levels(cls, v):
        if len(v) > 4:
            raise ValueError("At most 4 levels")
        return v

    @model_validator(mode="after")
    def check_sizes(self):
        if not 300 <= self.leaf_chars <= 20000:
            raise ValueError("leaf_chars must be between 300 and 20000")
        if not 0 <= self.min_chars < self.leaf_chars:
            raise ValueError("min_chars must be between 0 and leaf_chars")
        if not 2 <= self.fanout <= 16:
            raise ValueError("fanout must be between 2 and 16")
        return self


def _line_starts(text, start, end, pattern):
    regex = re.compile(pattern)
    cuts, pos = [], start
    while pos < end:
        newline = text.find("\n", pos, end)
        line_end = end if newline < 0 else newline
        if regex.match(text, pos, line_end):
            cuts.append(pos)
        pos = line_end + 1
    return cuts


def _cuts(text, start, end, splitter):
    """Positions strictly inside (start, end) where this splitter cuts."""
    if splitter.kind == "heading":
        cuts = _line_starts(text, start, end, splitter.regex)
    elif splitter.kind == "page_break":
        if splitter.regex:
            cuts = _line_starts(text, start, end, splitter.regex)
        else:
            cuts = [start + m.end() for m in re.finditer(r"\f", text[start:end])]
    elif splitter.kind == "blank_line":
        cuts = [start + m.end() for m in BLANK_LINE.finditer(text[start:end])]
    else:
        cuts, pos = [], start
        while end - pos > splitter.chars:
            target = pos + splitter.chars
            space = text.rfind(" ", pos + splitter.chars // 2, target)
            pos = space + 1 if space > pos else target
            cuts.append(pos)
    return sorted({c for c in cuts if start < c < end})


def _segments(start, end, cuts, min_chars):
    bounds = [start] + cuts + [end]
    segments = [[a, b] for a, b in zip(bounds, bounds[1:]) if b > a]
    merged = []
    for seg in segments:
        if merged and seg[1] - seg[0] < min_chars:
            merged[-1][1] = seg[1]
        else:
            merged.append(seg)
    if len(merged) > 1 and merged[0][1] - merged[0][0] < min_chars:
        merged[1][0] = merged[0][0]
        merged.pop(0)
    return [tuple(s) for s in merged]


def _leaf_cuts(text, start, end, leaf_chars):
    cuts, pos = [], start
    while end - pos > leaf_chars:
        window = text[pos + leaf_chars // 2: pos + leaf_chars]
        boundary = None
        for m in SENTENCE_END.finditer(window):
            boundary = pos + leaf_chars // 2 + m.end()
        if boundary is None:
            space = text.rfind(" ", pos + leaf_chars // 2, pos + leaf_chars)
            boundary = space + 1 if space > pos else pos + leaf_chars
        cuts.append(boundary)
        pos = boundary
    return [c for c in cuts if start < c < end]


def _title(text, start, end):
    first = text[start:min(end, start + 200)].strip().split("\n", 1)[0].strip()
    return first[:80] or "Region"


def build_spec_index(text: str, spec: IndexSpec) -> SemanticIndex:
    """Build a SemanticIndex for `text` from `spec`. Deterministic; no model calls."""
    nodes, counter = {}, [0]

    def new_id():
        counter[0] += 1
        return f"N{counter[0]}"

    def tree(start, end, level):
        """Nested (start, end, children) tuples for [start, end)."""
        while level < len(spec.levels):
            segments = _segments(start, end, _cuts(text, start, end, spec.levels[level]), spec.min_chars)
            if len(segments) > 1:
                return (start, end, [tree(a, b, level + 1) for a, b in segments])
            level += 1  # this level does not split this region; try the next one
        if end - start > spec.leaf_chars:
            bounds = [start] + _leaf_cuts(text, start, end, spec.leaf_chars) + [end]
            leaves = [(a, b, []) for a, b in zip(bounds, bounds[1:]) if b > a]
            if len(leaves) > 1:
                return (start, end, leaves)
        return (start, end, [])

    def group(children):
        """Contiguous groups so that no node has more than spec.fanout children."""
        if len(children) <= spec.fanout:
            return children
        parts = balanced_partition([(c[0], c[1]) for c in children], spec.fanout)
        grouped, i = [], 0
        for part in parts:
            members = children[i:i + len(part)]
            i += len(part)
            grouped.append(members[0] if len(members) == 1 else (members[0][0], members[-1][1], group(members)))
        return grouped

    def emit(node, parent, node_id=None):
        start, end, children = node
        node_id = node_id or new_id()
        child_ids = [emit(child, node_id) for child in group(children)] if children else []
        nodes[node_id] = IndexNode(node_id, start, end, _title(text, start, end), tuple(child_ids), parent)
        return node_id

    if not text:
        nodes["ROOT"] = IndexNode("ROOT", 0, 0, "Document", (), None)
    else:
        emit(tree(0, len(text), 0), None, "ROOT")
    index = SemanticIndex(text, "ROOT", nodes, policy="agent-spec",
                          metadata={"spec": spec.model_dump()})
    index.validate()
    return index


def index_stats(index: SemanticIndex) -> dict:
    leaves = [n.end - n.start for n in index.nodes.values() if not n.children]
    depth = {}

    def walk(node_id, d):
        depth[node_id] = d
        for child in index.nodes[node_id].children:
            walk(child, d + 1)

    walk(index.root_id, 0)
    leaves.sort()
    return {"chars": len(index.text), "nodes": len(index.nodes), "leaves": len(leaves),
            "depth": max(depth.values()), "leaf_chars_median": int(median(leaves)) if leaves else 0,
            "leaf_chars_max": leaves[-1] if leaves else 0}


def corpus_profile(texts: List[str], samples: int = 3, excerpt_chars: int = 1200, example_lines: int = 8) -> dict:
    """Label-free structure profile of a corpus for choosing an IndexSpec."""
    texts = [t for t in texts if isinstance(t, str)]
    if not texts:
        return {}
    lengths = sorted(len(t) for t in texts)
    q = lambda p: lengths[int(p * (len(lengths) - 1))]
    per_style = {}
    styles = [("heading", n, p, d) for n, (p, d) in HEADING_STYLES.items()] + \
             [("page_break", n, p, d) for n, (p, d) in PAGE_STYLES.items()]
    for kind, name, pattern, description in styles:
        m = line_matches(texts, pattern)
        per_style[name] = {"kind": kind, "description": description,
                           "matches_median_per_doc": m["matches_median"], "docs_with_any": m["docs_with_any"],
                           "share_of_lines_median": m["line_share_median"],
                           "section_chars_median": m["section_chars_median"],
                           "example_lines": m["examples"][:example_lines]}
    paragraphs = [len(p) for t in texts for p in BLANK_LINE.split(t) if p.strip()]
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    picks = sorted({order[int(f * (len(order) - 1))] for f in (0.25, 0.5, 0.75)})[:samples]
    excerpts = []
    for i in picks:
        t = texts[i]
        middle = max(0, len(t) // 2 - excerpt_chars // 3)
        excerpts.append({"chars": len(t), "start": t[:excerpt_chars],
                         "middle": t[middle:middle + excerpt_chars * 2 // 3]})
    return {
        "docs": len(texts),
        "chars": {"p10": q(0.1), "median": q(0.5), "p90": q(0.9), "max": lengths[-1]},
        "form_feeds_median_per_doc": int(median(t.count("\f") for t in texts)),
        "paragraphs": {"median_per_doc": int(median(len([p for p in BLANK_LINE.split(t) if p.strip()]) for t in texts)),
                       "median_chars": int(median(paragraphs)) if paragraphs else 0},
        "styles": per_style,
        "excerpts": excerpts,
    }


def line_matches(texts: List[str], pattern: str, examples: int = 10) -> dict:
    """How often `pattern` matches line starts across texts, and example matched lines."""
    regex = re.compile(pattern)
    counts, shares, sections, found = [], [], [], []
    for t in texts:
        lines = [line for line in t.split("\n") if line.strip()]
        matched = [line.strip() for line in lines if regex.match(line)]
        counts.append(len(matched))
        shares.append(len(matched) / max(1, len(lines)))
        sections.append(len(t) / (len(matched) + 1))
        found.extend(matched[::max(1, len(matched) // 2)][:2])   # spread through each document
    return {"matches_median": int(median(counts)), "docs_with_any": sum(c > 0 for c in counts),
            "line_share_median": round(median(shares), 4), "section_chars_median": int(median(sections)),
            "examples": [e[:80] for e in found[:examples]]}


def check_spec(spec: IndexSpec, samples: List[str]) -> dict:
    """Build `spec` on sample documents and return its shape, or raise ValueError with a
    reason (and the lines a pattern actually matched) that the designer can act on."""
    if not samples:
        return {}
    levels = []
    for i, level in enumerate(spec.levels):
        if level.kind not in ("heading", "page_break") or not level.regex:
            continue
        m = line_matches(samples, level.regex)
        shown = "; ".join(repr(e) for e in m["examples"]) or "none"
        name = level.style or level.pattern
        if m["docs_with_any"] == 0:
            raise ValueError(f"Level {i} ({name!r}) matches no line in any of {len(samples)} sample documents")
        if level.kind == "heading" and m["line_share_median"] > MAX_HEADING_LINE_SHARE:
            raise ValueError(f"Level {i} ({name!r}) matches {m['line_share_median']:.0%} of non-empty lines "
                             f"(limit {MAX_HEADING_LINE_SHARE:.0%}); it is selecting ordinary lines, not "
                             f"headings. Lines it matched: {shown}")
        min_section = MIN_SECTION_CHARS if level.kind == "heading" else MIN_PAGE_CHARS
        if m["section_chars_median"] < min_section:
            raise ValueError(f"Level {i} ({name!r}) creates sections of only {m['section_chars_median']} chars "
                             f"on average (minimum {min_section}). Lines it matched: {shown}")
        levels.append({"level": i, "splitter": name, "docs_with_matches": f"{m['docs_with_any']}/{len(samples)}",
                       "matches_median_per_doc": m["matches_median"],
                       "section_chars_median": m["section_chars_median"], "example_lines": m["examples"][:5]})
    stats = [index_stats(build_spec_index(t, spec)) for t in samples]
    leaves = median(s["leaves"] for s in stats)
    leaf_size = median(s["leaf_chars_median"] for s in stats)
    if leaf_size < 150:
        raise ValueError(f"Median leaf is only {leaf_size} chars; raise min_chars or use coarser levels")
    if leaves > 5000:
        raise ValueError(f"Median document has {leaves} leaves; use coarser levels or larger leaf_chars")
    return {"sample_docs": len(stats), "leaves_median": int(leaves),
            "depth_median": int(median(s["depth"] for s in stats)), "leaf_chars_median": int(leaf_size),
            "leaf_chars_max": max(s["leaf_chars_max"] for s in stats), "levels": levels}
