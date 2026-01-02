from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import faiss
import numpy as np

from utils.embedding_config import EmbeddingConfig, EmbeddingProvider
from utils.embedding_factory import EmbeddingFactory


@dataclass
class FinStdHit:
    term: str
    label: Optional[str]
    distance: float


class FinStdService:
    """金融术语标准化：将输入 term 在向量库中检索到最相近的标准术语。"""

    def __init__(
        self,
        provider: str = "huggingface",
        model: str = "BAAI/bge-m3",
        index_path: str = "db/fin_terms_bge_m3.faiss",
        meta_path: str = "db/fin_terms_meta.jsonl",
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

        self.index_path = index_path
        self.meta_path = meta_path

        self.index = faiss.read_index(self.index_path)
        self._meta = self._load_meta(self.meta_path)

    @staticmethod
    def _load_meta(meta_path: str) -> List[Dict]:
        import json

        items: List[Dict] = []
        with open(meta_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                items.append(json.loads(line))
        return items

    def search(self, query: str, limit: int = 5) -> List[Dict]:
        v = self.embedding.embed_query(query)
        q = np.asarray([v], dtype="float32")
        faiss.normalize_L2(q)

        distances, indices = self.index.search(q, k=limit)
        hits: List[Dict] = []
        for dist, idx in zip(distances[0].tolist(), indices[0].tolist()):
            if idx < 0 or idx >= len(self._meta):
                continue
            meta = self._meta[idx]
            hits.append({"term": meta.get("term"), "label": meta.get("label"), "distance": float(dist)})
        return hits
