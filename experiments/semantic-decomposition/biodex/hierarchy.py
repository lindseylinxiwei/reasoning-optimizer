"""Deterministic, label-only hierarchy for bounded Jev routing questions.

This is an embedding-derived hierarchy, not a biomedical ontology. Every
representative indexes an existing official label; no category names are invented.
"""
from __future__ import annotations

import math
import warnings

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits


VERSION = "embedding-hierarchy-v1"


def _representatives(indices, vectors, count):
    """Centroid medoid first, then farthest-first cosine coverage."""
    local = vectors[indices]
    centroid = local.mean(axis=0)
    center_norm = float(np.linalg.norm(centroid))
    if center_norm:
        centroid /= center_norm
    first = int(np.argmax(local @ centroid))
    selected = [first]
    # For normalized nonzero vectors, minimum squared distance corresponds to
    # maximum cosine similarity. Ties follow the original vocabulary order.
    nearest_similarity = local @ local[first]
    nearest_similarity[first] = np.inf
    while len(selected) < min(count, len(indices)):
        next_index = int(np.argmin(nearest_similarity))
        selected.append(next_index)
        nearest_similarity = np.maximum(nearest_similarity, local @ local[next_index])
        nearest_similarity[selected] = np.inf
    return [int(indices[i]) for i in selected]


def build_hierarchy(labels, vectors, leaf_size=96, branching=12,
                    representatives=16, seed=7):
    """Build a JSON-serializable tree without documents or ground truth.

    Children are ordered by their smallest vocabulary index. Repeated/zero
    vectors are valid: a collapsed split falls back to stable, even partitions,
    ensuring every terminal node respects ``leaf_size``.
    """
    if not labels:
        raise ValueError("labels must be nonempty")
    if leaf_size < 1 or branching < 2 or representatives < 1:
        raise ValueError("leaf_size/representatives must be positive; branching >= 2")
    values = np.array(vectors, dtype=np.float32, copy=True)
    if values.ndim != 2 or values.shape[0] != len(labels) or not values.shape[1]:
        raise ValueError("vectors must have shape (len(labels), positive dimensions)")
    if not np.isfinite(values).all():
        raise ValueError("vectors must be finite")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    values /= np.where(norms > 0, norms, 1.0)
    tree = {
        "version": VERSION,
        "config": {
            "leaf_size": int(leaf_size), "branching": int(branching),
            "representatives": int(representatives), "seed": int(seed),
            "clustering": "MiniBatchKMeans", "normalized_vectors": True,
            "representative_selection": "centroid_medoid_then_farthest_first",
        },
        "root": "N00000", "nodes": {},
    }

    def visit(indices):
        node_id = f"N{len(tree['nodes']):05d}"
        node = {
            "id": node_id, "label_indices": indices.tolist(), "children": [],
            "representative_indices": _representatives(indices, values, representatives),
        }
        tree["nodes"][node_id] = node
        if len(indices) <= leaf_size:
            return node_id
        cluster_count = min(branching, math.ceil(len(indices) / leaf_size))
        local = values[indices]
        if np.all(local == local[0]):
            groups = [indices]
        else:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ConvergenceWarning)
                assignment = MiniBatchKMeans(
                    n_clusters=cluster_count, random_state=seed, n_init=3,
                    max_iter=80, batch_size=min(1024, len(indices)),
                    reassignment_ratio=0.0,
                ).fit_predict(local)
            groups = [indices[assignment == k] for k in range(cluster_count)]
            groups = [group for group in groups if len(group)]
        if len(groups) < 2:
            groups = list(np.array_split(indices, cluster_count))
            node["split_fallback"] = "stable_even_partition"
        # Each group preserves original vocabulary order.
        groups.sort(key=lambda group: int(group[0]))
        node["children"] = [visit(group) for group in groups]
        return node_id

    with threadpool_limits(limits=1):
        tree["root"] = visit(np.arange(len(labels), dtype=np.int64))
    validate_hierarchy(tree, len(labels))
    return tree


def validate_hierarchy(tree, nlabels):
    """Raise ValueError for malformed topology, overlap, or missing labels."""
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    require(isinstance(nlabels, int) and nlabels > 0, "nlabels must be positive")
    require(tree.get("version") == VERSION, "unsupported hierarchy version")
    nodes = tree.get("nodes", {})
    root = tree.get("root")
    config = tree.get("config", {})
    require(root in nodes, "missing root")
    leaf_size = config.get("leaf_size", 0)
    branching = config.get("branching", 0)
    rep_count = config.get("representatives", 0)
    require(leaf_size >= 1 and branching >= 2 and rep_count >= 1, "invalid config")
    visited, active, leaves = set(), set(), []

    def walk(node_id):
        require(node_id in nodes, f"unknown child {node_id}")
        require(node_id not in active, f"cycle at {node_id}")
        require(node_id not in visited, f"multiple parents for {node_id}")
        active.add(node_id)
        visited.add(node_id)
        node = nodes[node_id]
        require(node.get("id") == node_id, f"id mismatch at {node_id}")
        indices = node.get("label_indices", [])
        require(bool(indices), f"empty node {node_id}")
        require(all(isinstance(i, int) and not isinstance(i, bool)
                    and 0 <= i < nlabels for i in indices), f"invalid indices at {node_id}")
        require(indices == sorted(set(indices)), f"unordered/duplicate indices at {node_id}")
        members = set(indices)
        reps = node.get("representative_indices", [])
        require(len(reps) == min(rep_count, len(indices)), f"wrong representative count at {node_id}")
        require(len(reps) == len(set(reps)) and set(reps) <= members,
                f"invalid representatives at {node_id}")
        children = node.get("children", [])
        require(len(children) == len(set(children)), f"duplicate children at {node_id}")
        if children:
            require(2 <= len(children) <= branching, f"invalid branching at {node_id}")
            descendants = set()
            for child in children:
                child_members = walk(child)
                require(not descendants.intersection(child_members), f"overlapping children at {node_id}")
                descendants.update(child_members)
            require(descendants == members, f"child coverage mismatch at {node_id}")
        else:
            require(len(indices) <= leaf_size, f"oversized leaf {node_id}")
            leaves.extend(indices)
        active.remove(node_id)
        return members

    coverage = walk(root)
    require(coverage == set(range(nlabels)), "root does not cover the vocabulary")
    require(visited == set(nodes), "orphan nodes")
    require(sorted(leaves) == list(range(nlabels)), "terminal leaves must cover each label exactly once")
    return True
