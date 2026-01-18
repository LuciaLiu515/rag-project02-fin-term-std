from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class HierarchicalIndexData:
    parents: List[Dict]
    children: List[Dict]


def build_hierarchical_index_data(
    terms: List[Dict],
    *,
    parent_size: int = 200,
) -> HierarchicalIndexData:
    """Build hierarchical index data.

    Implements "02-构建有层次的索引":
      - Parent nodes: coarse summaries (here: a block of N terms).
      - Child nodes: fine-grained items assigned to a parent.

    For a term dictionary CSV, we don't have long documents, so we treat blocks
    of terms as a "topic cluster". Parents store a compact summary string.

    Output shapes are JSON friendly and meant to be embedded separately.
    """

    if parent_size <= 0:
        parent_size = 200

    parents: List[Dict] = []
    children: List[Dict] = []

    parent_id = 0
    for start in range(0, len(terms), parent_size):
        block = terms[start : start + parent_size]
        end = start + len(block) - 1

        # Simple summary: take top-k terms in the block.
        # (In real docs, you'd use LLM summarization.)
        sample = [b.get("term", "") for b in block[: min(30, len(block))]]
        summary = " | ".join([s for s in sample if s])

        parents.append(
            {
                "parent_id": parent_id,
                "range": [start, end],
                "summary": summary,
            }
        )

        for local_i, item in enumerate(block):
            children.append(
                {
                    "child_id": start + local_i,
                    "parent_id": parent_id,
                    "term": item.get("term"),
                    "label": item.get("label"),
                }
            )

        parent_id += 1

    return HierarchicalIndexData(parents=parents, children=children)
