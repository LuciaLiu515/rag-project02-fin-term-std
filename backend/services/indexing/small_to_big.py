from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import faiss
import numpy as np


@dataclass
class SmallToBigResult:
    small_chunks: List[Dict]
    big_chunks: List[Dict]


def _normalize(x: np.ndarray) -> np.ndarray:
    x = x.astype("float32", copy=False)
    faiss.normalize_L2(x)
    return x


def build_small_to_big_chunks(
    terms: List[Dict],
    *,
    window: int = 5,
) -> SmallToBigResult:
    """Build a small-to-big context mapping.

    This implements the idea "01-从小块到大上下文":
    - We index *small chunks* (here: individual term records).
    - Each small chunk keeps a pointer to a *big chunk* that provides larger context.

    For fin-term-std CSV, each record is small already, so we create "big" by
    concatenating a sliding window of neighboring terms.

    Args:
        terms: list of {term, label, ...}
        window: number of neighbors on each side to form big context.

    Returns:
        SmallToBigResult with:
          - small_chunks: each includes {term,label,big_id}
          - big_chunks: each includes {big_id,content,start,end}
    """

    if window < 0:
        window = 0

    big_chunks: List[Dict] = []
    # Prebuild big chunks for each position
    for i in range(len(terms)):
        start = max(0, i - window)
        end = min(len(terms), i + window + 1)
        # Simple context: a list of terms and optional labels
        lines = []
        for j in range(start, end):
            t = terms[j].get("term", "")
            lb = terms[j].get("label", "")
            if lb:
                lines.append(f"{t} ({lb})")
            else:
                lines.append(str(t))
        big_chunks.append(
            {
                "big_id": i,
                "content": "\n".join(lines),
                "range": [start, end - 1],
            }
        )

    small_chunks: List[Dict] = []
    for i, item in enumerate(terms):
        small_chunks.append(
            {
                "term": item.get("term"),
                "label": item.get("label"),
                "big_id": i,
            }
        )

    return SmallToBigResult(small_chunks=small_chunks, big_chunks=big_chunks)
