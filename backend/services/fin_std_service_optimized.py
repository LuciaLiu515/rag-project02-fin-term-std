from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional

import faiss
import numpy as np

from utils.embedding_config import EmbeddingConfig, EmbeddingProvider
from utils.embedding_factory import EmbeddingFactory


@dataclass
class FinStdHit:
    term: str
    label: Optional[str]
    distance: float
    meta: Dict


IndexMode = Literal["baseline", "small_to_big", "hierarchical", "multi"]


class FinStdServiceOptimized:
    """Optimized retrieval strategies for fin-term-std.

    This keeps the original project's behavior available (baseline), and adds:
      1) small_to_big: search on small index, then return big context
      2) hierarchical: search parent, then search children within matched parents
      3) multi: search on multi-representation index, then map back to canonical term

    All indexes are FAISS cosine (normalized + InnerProduct).
    """

    def __init__(
        self,
        *,
        provider: str = "huggingface",
        model: str = "BAAI/bge-m3",
        # baseline
        index_path: str = "db/fin_terms_bge_m3.faiss",
        meta_path: str = "db/fin_terms_meta.jsonl",
        # optimized
        small_index_path: str = "db/fin_terms_small.faiss",
        small_meta_path: str = "db/fin_terms_small_meta.jsonl",
        big_index_path: str = "db/fin_terms_big.faiss",
        big_meta_path: str = "db/fin_terms_big_meta.jsonl",
        parent_index_path: str = "db/fin_terms_parent.faiss",
        parent_meta_path: str = "db/fin_terms_parent_meta.jsonl",
        child_index_path: str = "db/fin_terms_child.faiss",
        child_meta_path: str = "db/fin_terms_child_meta.jsonl",
        multi_index_path: str = "db/fin_terms_multi.faiss",
        multi_meta_path: str = "db/fin_terms_multi_meta.jsonl",
    ):
        provider_mapping = {
            "openai": EmbeddingProvider.OPENAI,
            "huggingface": EmbeddingProvider.HUGGINGFACE,
        }
        embedding_provider = provider_mapping.get(provider.lower())
        if embedding_provider is None:
            raise ValueError(f"Unsupported provider: {provider}")

        config = EmbeddingConfig(provider=embedding_provider, model_name=model)
        self.embedding = EmbeddingFactory.create_embedding_function(config)

        self.baseline_index = faiss.read_index(index_path)
        self.baseline_meta = self._load_jsonl(meta_path)

        # optimized
        self.small_index = faiss.read_index(small_index_path)
        self.small_meta = self._load_jsonl(small_meta_path)

        self.big_index = faiss.read_index(big_index_path)
        self.big_meta = self._load_jsonl(big_meta_path)

        self.parent_index = faiss.read_index(parent_index_path)
        self.parent_meta = self._load_jsonl(parent_meta_path)

        self.child_index = faiss.read_index(child_index_path)
        self.child_meta = self._load_jsonl(child_meta_path)

        self.multi_index = faiss.read_index(multi_index_path)
        self.multi_meta = self._load_jsonl(multi_meta_path)

        # for quick lookup
        self._big_by_id = {m.get("big_id"): m for m in self.big_meta}

    @staticmethod
    def _load_jsonl(path: str) -> List[Dict]:
        import json

        items: List[Dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

    def _embed(self, query: str) -> np.ndarray:
        v = self.embedding.embed_query(query)
        q = np.asarray([v], dtype="float32")
        faiss.normalize_L2(q)
        return q

    def search(self, query: str, *, limit: int = 5, mode: IndexMode = "baseline") -> List[Dict]:
        if mode == "baseline":
            return self._search_baseline(query, limit=limit)
        if mode == "small_to_big":
            return self._search_small_to_big(query, limit=limit)
        if mode == "hierarchical":
            return self._search_hierarchical(query, limit=limit)
        if mode == "multi":
            return self._search_multi(query, limit=limit)
        raise ValueError(f"Unknown mode: {mode}")

    def _search_baseline(self, query: str, *, limit: int) -> List[Dict]:
        q = self._embed(query)
        distances, indices = self.baseline_index.search(q, k=limit)
        hits: List[Dict] = []
        for dist, idx in zip(distances[0].tolist(), indices[0].tolist()):
            if idx < 0 or idx >= len(self.baseline_meta):
                continue
            meta = self.baseline_meta[idx]
            hits.append({"term": meta.get("term"), "label": meta.get("label"), "distance": float(dist)})
        return hits

    def _search_small_to_big(self, query: str, *, limit: int) -> List[Dict]:
        q = self._embed(query)
        distances, indices = self.small_index.search(q, k=limit)
        hits: List[Dict] = []
        for dist, idx in zip(distances[0].tolist(), indices[0].tolist()):
            if idx < 0 or idx >= len(self.small_meta):
                continue
            small = self.small_meta[idx]
            big = self._big_by_id.get(small.get("big_id"), {})
            hits.append(
                {
                    "term": small.get("term"),
                    "label": small.get("label"),
                    "distance": float(dist),
                    "big_context": big.get("content"),
                    "big_range": big.get("range"),
                }
            )
        return hits

    def _search_hierarchical(self, query: str, *, limit: int) -> List[Dict]:
        # Step1: search parents
        q = self._embed(query)
        parent_dist, parent_idx = self.parent_index.search(q, k=min(5, max(1, limit)))
        parent_ids = []
        parent_hits = []
        for dist, idx in zip(parent_dist[0].tolist(), parent_idx[0].tolist()):
            if idx < 0 or idx >= len(self.parent_meta):
                continue
            p = self.parent_meta[idx]
            pid = p.get("parent_id")
            if pid is None:
                continue
            parent_ids.append(pid)
            parent_hits.append({"parent_id": pid, "parent_distance": float(dist), "parent_range": p.get("range")})

        if not parent_ids:
            return []

        # Step2: search children globally, then filter by parent_id
        # (For minimal changes we keep a single FAISS index; filtering is done in Python.)
        child_dist, child_idx = self.child_index.search(q, k=max(50, limit * 10))
        hits: List[Dict] = []
        for dist, idx in zip(child_dist[0].tolist(), child_idx[0].tolist()):
            if idx < 0 or idx >= len(self.child_meta):
                continue
            c = self.child_meta[idx]
            if c.get("parent_id") not in parent_ids:
                continue
            hits.append(
                {
                    "term": c.get("term"),
                    "label": c.get("label"),
                    "distance": float(dist),
                    "parent_id": c.get("parent_id"),
                }
            )
            if len(hits) >= limit:
                break

        return hits

    def _search_multi(self, query: str, *, limit: int) -> List[Dict]:
        q = self._embed(query)
        distances, indices = self.multi_index.search(q, k=max(20, limit * 5))

        # Deduplicate by canonical term_id
        hits: List[Dict] = []
        seen = set()
        for dist, idx in zip(distances[0].tolist(), indices[0].tolist()):
            if idx < 0 or idx >= len(self.multi_meta):
                continue
            r = self.multi_meta[idx]
            term_id = r.get("term_id")
            if term_id in seen:
                continue
            seen.add(term_id)
            hits.append(
                {
                    "term": r.get("canonical_term"),
                    "label": r.get("label"),
                    "distance": float(dist),
                    "matched_representation": r.get("representation"),
                    "rep_type": r.get("rep_type"),
                }
            )
            if len(hits) >= limit:
                break

        return hits
