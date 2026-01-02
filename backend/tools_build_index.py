"""Build a local FAISS vector index from 万条金融标准术语.csv.

Why FAISS?
- On Windows, pymilvus Milvus Lite backend may be unavailable.
- FAISS provides a simple local-first index (.faiss) with a meta file.

Input CSV format (tolerant):
- column 1: term
- column 2: label (optional)

Outputs:
- backend/db/fin_terms_bge_m3.faiss
- backend/db/fin_terms_meta.jsonl
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import dotenv
import faiss
import numpy as np

from utils.embedding_config import EmbeddingConfig, EmbeddingProvider
from utils.embedding_factory import EmbeddingFactory


dotenv.load_dotenv()


_HERE = os.path.dirname(os.path.abspath(__file__))


def _abs_from_here(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_HERE, path)


@dataclass
class BuildOptions:
    provider: str = "huggingface"
    model: str = "BAAI/bge-m3"
    csv_path: str = "data/万条金融标准术语.csv"
    index_path: str = "db/fin_terms_bge_m3.faiss"
    meta_path: str = "db/fin_terms_meta.jsonl"
    batch_size: int = 512


def _iter_terms(csv_path: str) -> Iterable[Tuple[str, str]]:
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            term = (row[0] or "").strip()
            label = (row[1] or "").strip() if len(row) > 1 else ""
            if not term:
                continue
            yield term, label


def build_index(opts: BuildOptions) -> None:
    provider_mapping = {
        "openai": EmbeddingProvider.OPENAI,
        "huggingface": EmbeddingProvider.HUGGINGFACE,
    }
    embedding_provider = provider_mapping.get(opts.provider.lower())
    if embedding_provider is None:
        raise ValueError(f"Unsupported provider: {opts.provider}")

    embedding = EmbeddingFactory.create_embedding_function(
        EmbeddingConfig(provider=embedding_provider, model_name=opts.model)
    )

    csv_path = _abs_from_here(opts.csv_path)
    index_path = _abs_from_here(opts.index_path)
    meta_path = _abs_from_here(opts.meta_path)

    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    items = list(_iter_terms(csv_path))
    if not items:
        raise ValueError(f"No terms found in {csv_path}")

    vectors: List[List[float]] = []
    metas: List[dict] = []

    for start in range(0, len(items), opts.batch_size):
        batch = items[start : start + opts.batch_size]
        texts = [t for (t, _label) in batch]
        batch_vecs = embedding.embed_documents(texts)
        vectors.extend(batch_vecs)
        metas.extend(
            {
                "term": batch[i][0],
                "label": batch[i][1],
            }
            for i in range(len(batch))
        )

    x = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(x)  # cosine similarity via inner product on normalized vectors

    dim = x.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(x)

    faiss.write_index(index, index_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        for m in metas:
            f.write(json.dumps(m, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    build_index(BuildOptions())
