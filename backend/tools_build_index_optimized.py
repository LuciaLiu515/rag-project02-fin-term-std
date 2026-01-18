"""Build optimized FAISS indexes for fin-term-std.

This file implements key ideas from:
https://github.com/huangjia2019/rag-in-action/tree/master/06-%E7%B4%A2%E5%BC%95%E4%BC%98%E5%8C%96-Indexing

Implemented (minimal changes to existing project):
1) 01-从小块到大上下文 (small-to-big)
2) 02-构建有层次的索引 (hierarchical)
3) 03-构建多表示的索引 (multi-representation)

Outputs (written under backend/db by default):
- fin_terms_small.faiss + fin_terms_small_meta.jsonl
- fin_terms_big.faiss + fin_terms_big_meta.jsonl
- fin_terms_parent.faiss + fin_terms_parent_meta.jsonl
- fin_terms_child.faiss + fin_terms_child_meta.jsonl
- fin_terms_multi.faiss + fin_terms_multi_meta.jsonl

All indexes use cosine similarity via L2-normalization + IndexFlatIP.

Usage (PowerShell, after installing deps):
  python tools_build_index_optimized.py

Note: this is intentionally separate from tools_build_index.py to keep the original workflow intact.
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

from services.indexing.hierarchical import build_hierarchical_index_data
from services.indexing.multi_representation import build_multi_representation_data
from services.indexing.small_to_big import build_small_to_big_chunks
from utils.embedding_config import EmbeddingConfig, EmbeddingProvider
from utils.embedding_factory import EmbeddingFactory


dotenv.load_dotenv()

_HERE = os.path.dirname(os.path.abspath(__file__))


def _abs_from_here(path: str) -> str:
    return path if os.path.isabs(path) else os.path.join(_HERE, path)


def _write_jsonl(path: str, rows: List[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


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


def _build_faiss_index(vectors: List[List[float]]) -> faiss.Index:
    x = np.asarray(vectors, dtype="float32")
    faiss.normalize_L2(x)
    dim = x.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(x)
    return index


@dataclass
class BuildOptionsOptimized:
    provider: str = "huggingface"
    model: str = "BAAI/bge-m3"
    csv_path: str = "data/万条金融标准术语.csv"
    db_dir: str = "db"

    # small-to-big
    window: int = 5

    # hierarchical
    parent_size: int = 200


def build_indexes(opts: BuildOptionsOptimized) -> None:
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
    db_dir = _abs_from_here(opts.db_dir)
    os.makedirs(db_dir, exist_ok=True)

    terms = [{"term": t, "label": lb} for t, lb in _iter_terms(csv_path)]
    if not terms:
        raise ValueError(f"No terms found in {csv_path}")

    # 01) small-to-big
    s2b = build_small_to_big_chunks(terms, window=opts.window)

    # Small vectors: embed term only (consistent with original project)
    small_texts = [c["term"] for c in s2b.small_chunks]
    small_vecs = embedding.embed_documents(small_texts)
    small_index = _build_faiss_index(small_vecs)

    faiss.write_index(small_index, os.path.join(db_dir, "fin_terms_small.faiss"))
    _write_jsonl(os.path.join(db_dir, "fin_terms_small_meta.jsonl"), s2b.small_chunks)

    # Big vectors: embed big content, meta includes big_id + content
    big_texts = [b["content"] for b in s2b.big_chunks]
    big_vecs = embedding.embed_documents(big_texts)
    big_index = _build_faiss_index(big_vecs)

    faiss.write_index(big_index, os.path.join(db_dir, "fin_terms_big.faiss"))
    _write_jsonl(os.path.join(db_dir, "fin_terms_big_meta.jsonl"), s2b.big_chunks)

    # 02) hierarchical
    hier = build_hierarchical_index_data(terms, parent_size=opts.parent_size)

    parent_texts = [p["summary"] for p in hier.parents]
    parent_vecs = embedding.embed_documents(parent_texts)
    parent_index = _build_faiss_index(parent_vecs)

    faiss.write_index(parent_index, os.path.join(db_dir, "fin_terms_parent.faiss"))
    _write_jsonl(os.path.join(db_dir, "fin_terms_parent_meta.jsonl"), hier.parents)

    child_texts = [c["term"] for c in hier.children]
    child_vecs = embedding.embed_documents(child_texts)
    child_index = _build_faiss_index(child_vecs)

    faiss.write_index(child_index, os.path.join(db_dir, "fin_terms_child.faiss"))
    _write_jsonl(os.path.join(db_dir, "fin_terms_child_meta.jsonl"), hier.children)

    # 03) multi-representation
    multi = build_multi_representation_data(terms)
    multi_texts = [r["representation"] for r in multi.reps]
    multi_vecs = embedding.embed_documents(multi_texts)
    multi_index = _build_faiss_index(multi_vecs)

    faiss.write_index(multi_index, os.path.join(db_dir, "fin_terms_multi.faiss"))
    _write_jsonl(os.path.join(db_dir, "fin_terms_multi_meta.jsonl"), multi.reps)


if __name__ == "__main__":
    build_indexes(BuildOptionsOptimized())
