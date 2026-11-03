"""Query-independent topic hierarchy from injected document embeddings.

No embedding client, task, query, annotation, or answer-candidate filter exists
here. Vectors influence contiguous grouping boundaries only; original text and
every source interval remain reachable. Deployment must charge the full document
embedding index cost even when one prepared vector set serves several methods.
"""
from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Mapping, Sequence

try:
    from .semantic_index import IndexNode, SemanticIndex, balanced_partition
except ImportError:
    from semantic_index import IndexNode, SemanticIndex, balanced_partition


def _span(window: Any) -> tuple[int, int]:
    if isinstance(window, Mapping):
        values = window.get("start"), window.get("end")
    elif hasattr(window, "start") and hasattr(window, "end"):
        values = window.start, window.end
    elif isinstance(window, (tuple, list)) and len(window) == 2:
        values = tuple(window)
    else:
        raise ValueError("Windows require exact original start/end offsets")
    if any(type(value) is not int for value in values):
        raise ValueError("Window offsets must be integers")
    return values


def _normalize(vector: Sequence[float]) -> tuple[float, ...]:
    if (not isinstance(vector, (tuple, list)) or not vector
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)):
        raise ValueError("Injected vectors must be nonempty finite numeric sequences")
    norm = math.hypot(*vector)
    if not math.isfinite(norm) or norm == 0:
        raise ValueError("Injected vector has zero or invalid magnitude")
    return tuple(value / norm for value in vector)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    return min(1.0, max(-1.0, math.fsum(a * b for a, b in zip(left, right))))


def _atomic_projection(text: str, windows: Sequence[Any], vectors: Sequence[Sequence[float]]):
    """Project overlapping source windows onto an exact disjoint partition."""
    if len(windows) != len(vectors) or (text and not windows):
        raise ValueError("Every source window requires one injected vector")
    records = [(_span(window), _normalize(vector)) for window, vector in zip(windows, vectors)]
    if records and len({len(vector) for _, vector in records}) != 1:
        raise ValueError("Injected vector dimensions differ")
    records = sorted(records, key=lambda value: value[0])
    events: dict[int, dict[str, list[int]]] = {}
    for i, ((start, end), _) in enumerate(records):
        if not 0 <= start < end <= len(text):
            raise ValueError("Invalid original-source window")
        events.setdefault(start, {"start": [], "end": []})["start"].append(i)
        events.setdefault(end, {"start": [], "end": []})["end"].append(i)
    if not text:
        return [], [], {"input_windows": 0, "atomic_regions": 0, "zero_mean_fallbacks": 0}
    cuts = sorted({0, len(text), *events})
    atoms, projected, active, zero_mean = [], [], set(), 0
    for start, end in zip(cuts, cuts[1:]):
        event = events.get(start, {"start": [], "end": []})
        active.difference_update(event["end"])
        active.update(event["start"])
        if not active:
            raise ValueError("Injected source windows do not cover the complete document")
        values = [records[i][1] for i in sorted(active)]
        if len(values) == 1:
            vector = values[0]
        else:
            mean = tuple(math.fsum(value[j] for value in values) / len(values) for j in range(len(values[0])))
            # Opposing vectors can cancel. Keep the first normalized window in
            # stable source order as an explicit, counted semantic fallback.
            if math.hypot(*mean) <= 1e-12:
                vector = values[0]
                zero_mean += 1
            else:
                vector = _normalize(mean)
        atoms.append((start, end))
        projected.append(vector)
    provenance = {
        "input_windows": len(records), "atomic_regions": len(atoms),
        "overlapping_windows": any(records[i][0][0] < records[i - 1][0][1] for i in range(1, len(records))),
        "zero_mean_fallbacks": zero_mean,
        "window_intervals_sha256": hashlib.sha256(json.dumps([span for span, _ in records], separators=(",", ":")).encode()).hexdigest(),
        "normalized_window_vectors_sha256": hashlib.sha256(json.dumps([vector for _, vector in records], separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
        "dimensions": len(records[0][1]),
    }
    return atoms, projected, provenance


def _topic_partition(atoms: Sequence[tuple[int, int]], similarities: Sequence[float],
                     fanout: int, balance_tolerance: float):
    """Minimize adjacent cut similarity within source-character balance limits.

    Dynamic programming chooses the whole set of cuts jointly, avoiding a greedy
    early semantic cut that leaves a tiny later child. If indivisible source units
    make the balance constraint infeasible, decrease child count; final fallback
    uses the common source-capacity partition and reports the relaxation.
    """
    count = len(atoms)
    prefix = [0]
    for start, end in atoms:
        prefix.append(prefix[-1] + end - start)
    max_groups = min(fanout, count)
    for groups in range(max_groups, 1, -1):
        target = prefix[-1] / groups
        lower, upper = target * (1 - balance_tolerance), target * (1 + balance_tolerance)
        # State value: (semantic cut cost, balance penalty, lexicographic cuts).
        states: dict[int, tuple[float, float, tuple[int, ...]]] = {0: (0.0, 0.0, ())}
        for group in range(1, groups + 1):
            following = {}
            for end in range(group, count - (groups - group) + 1):
                options = []
                for start, (cost, penalty, cuts) in states.items():
                    if start >= end:
                        continue
                    length = prefix[end] - prefix[start]
                    if not lower <= length <= upper:
                        continue
                    cut_cost = similarities[end - 1] if end < count else 0.0
                    options.append((cost + cut_cost, penalty + ((length - target) / target) ** 2,
                                    cuts + ((end,) if end < count else ())))
                if options:
                    following[end] = min(options)
            states = following
            if not states:
                break
        if count in states:
            best = states[count]
            cuts = (0, *best[2], count)
            return tuple(zip(cuts, cuts[1:])), {
                "requested_children": max_groups, "children": groups,
                "balance_relaxed": False, "semantic_cut_cost": best[0],
                "child_character_lengths": [prefix[b] - prefix[a] for a, b in zip(cuts, cuts[1:])],
            }
    fallback = balanced_partition(atoms, max_groups)
    cursor, groups = 0, []
    for values in fallback:
        groups.append((cursor, cursor + len(values)))
        cursor += len(values)
    return tuple(groups), {
        "requested_children": max_groups, "children": len(groups),
        "balance_relaxed": True, "semantic_cut_cost": None,
        "child_character_lengths": [prefix[b] - prefix[a] for a, b in groups],
    }


def build_topic_index(text: str, windows: Sequence[Any], vectors: Sequence[Sequence[float]],
                      fanout: int = 4, leaf_chars: int = 2400,
                      balance_tolerance: float = 0.5) -> SemanticIndex:
    """Build a contiguous source-topic tree with no query-specific admission.

    Source-window embeddings influence grouping boundaries, never text inclusion.
    `leaf_chars` is a stopping target: an indivisible provided atomic region may
    remain larger, and all region/source request paging belongs to the navigator.
    The input embedding index must itself cover the entire original source.
    """
    if not isinstance(text, str):
        raise TypeError("Original source must be text")
    if type(fanout) is not int or fanout < 2:
        raise ValueError("fanout must be an integer of at least two")
    if type(leaf_chars) is not int or leaf_chars < 32:
        raise ValueError("leaf_chars must be an integer of at least 32")
    if type(balance_tolerance) not in (int, float) or not math.isfinite(balance_tolerance) or not 0 <= balance_tolerance < 1:
        raise ValueError("balance_tolerance must be finite and within [0,1)")
    atoms, projected, provenance = _atomic_projection(text, windows, vectors)
    neighbors = [_cosine(left, right) for left, right in zip(projected, projected[1:])]
    nodes: dict[str, IndexNode] = {}
    kinds: dict[str, str] = {}
    partitions = []

    def build(first: int, last: int, parent: str | None) -> str:
        start, end = (atoms[first][0], atoms[last - 1][1]) if atoms else (0, 0)
        node_id = f"topic:{start}:{end}"
        children = ()
        leaf = last - first <= 1 or end - start <= leaf_chars
        kind = "root" if parent is None else ("source-window" if leaf else "embedding-topic-group")
        kinds[node_id] = kind
        title = "Document" if parent is None else f"Source region [{start},{end})"
        if not leaf:
            groups, diagnostics = _topic_partition(atoms[first:last], neighbors[first:last - 1], fanout, balance_tolerance)
            partitions.append({"node": node_id, **diagnostics})
            children = tuple(build(first + a, first + b, node_id) for a, b in groups)
        nodes[node_id] = IndexNode(node_id, start, end, title, children, parent)
        return node_id

    root_id = build(0, len(atoms), None)
    index = SemanticIndex(text, root_id, nodes, policy="embedding-topic", node_kinds=kinds,
                          metadata={
                              "query_independent": True, "source_preserved": True,
                              "classification": "embedding-informed contiguous topic hierarchy; boundaries remain heuristic",
                              "fanout": fanout, "leaf_chars": leaf_chars, "balance_tolerance": balance_tolerance,
                              "projection": "normalize window vectors; normalized mean over every window covering an atomic source region",
                              "partition_objective": "minimize sum of cosine similarity at child boundaries subject to character-balance limits",
                              "partition_tiebreak": "minimum squared length imbalance, then lexicographically earliest atomic boundaries",
                              "balance_fallback": "decrease child count; capacity partition if every balanced topic partition is infeasible",
                              "index_cost_policy": "charge full standalone source embedding cost; shared experimental vectors do not imply free deployment indexing",
                              "embedding_provider": "injected; provenance must be recorded by the experiment runner",
                              "projection_provenance": provenance, "partitions": partitions,
                          })
    leaves = index.leaves()
    for i, node in enumerate(leaves):
        index.links[node.id] = {"previous": (leaves[i - 1].id,) if i else (),
                                "next": (leaves[i + 1].id,) if i + 1 < len(leaves) else (),
                                "topic": ()}
    index.validate()
    return index
