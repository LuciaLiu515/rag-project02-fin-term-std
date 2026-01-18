from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional

from services.fin_std_service import FinStdService
from services.fin_std_service_optimized import FinStdServiceOptimized


app = FastAPI(title="Financial Terminology Standardization")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class EmbeddingOptions(BaseModel):
    provider: Literal["huggingface", "openai"] = Field(default="huggingface")
    model: str = Field(default="BAAI/bge-m3")
    indexPath: str = Field(default="db/fin_terms_bge_m3.faiss")
    metaPath: str = Field(default="db/fin_terms_meta.jsonl")


class FinStdInput(BaseModel):
    text: str
    topK: int = 5
    embeddingOptions: EmbeddingOptions = Field(default_factory=EmbeddingOptions)


class FinStdOptimizedInput(FinStdInput):
    indexMode: Literal["baseline", "small_to_big", "hierarchical", "multi"] = Field(default="baseline")

    # Paths for optimized indexes (optional; default to backend/db/*)
    smallIndexPath: str = Field(default="db/fin_terms_small.faiss")
    smallMetaPath: str = Field(default="db/fin_terms_small_meta.jsonl")
    bigIndexPath: str = Field(default="db/fin_terms_big.faiss")
    bigMetaPath: str = Field(default="db/fin_terms_big_meta.jsonl")
    parentIndexPath: str = Field(default="db/fin_terms_parent.faiss")
    parentMetaPath: str = Field(default="db/fin_terms_parent_meta.jsonl")
    childIndexPath: str = Field(default="db/fin_terms_child.faiss")
    childMetaPath: str = Field(default="db/fin_terms_child_meta.jsonl")
    multiIndexPath: str = Field(default="db/fin_terms_multi.faiss")
    multiMetaPath: str = Field(default="db/fin_terms_multi_meta.jsonl")


class FinStdBatchInput(BaseModel):
    texts: List[str]
    topK: int = 5
    embeddingOptions: EmbeddingOptions = Field(default_factory=EmbeddingOptions)


@app.post("/api/fin/std")
async def fin_std(input: FinStdInput):
    try:
        svc = FinStdService(
            provider=input.embeddingOptions.provider,
            model=input.embeddingOptions.model,
            index_path=input.embeddingOptions.indexPath,
            meta_path=input.embeddingOptions.metaPath,
        )
        return {
            "query": input.text,
            "results": svc.search(input.text, limit=input.topK),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fin/std/optimized")
async def fin_std_optimized(input: FinStdOptimizedInput):
    """Optimized retrieval endpoint.

    Implements:
      - small_to_big: return big context
      - hierarchical: parent->child
      - multi: multi-representation
    """
    try:
        svc = FinStdServiceOptimized(
            provider=input.embeddingOptions.provider,
            model=input.embeddingOptions.model,
            index_path=input.embeddingOptions.indexPath,
            meta_path=input.embeddingOptions.metaPath,
            small_index_path=input.smallIndexPath,
            small_meta_path=input.smallMetaPath,
            big_index_path=input.bigIndexPath,
            big_meta_path=input.bigMetaPath,
            parent_index_path=input.parentIndexPath,
            parent_meta_path=input.parentMetaPath,
            child_index_path=input.childIndexPath,
            child_meta_path=input.childMetaPath,
            multi_index_path=input.multiIndexPath,
            multi_meta_path=input.multiMetaPath,
        )
        return {
            "query": input.text,
            "indexMode": input.indexMode,
            "results": svc.search(input.text, limit=input.topK, mode=input.indexMode),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/fin/std/batch")
async def fin_std_batch(input: FinStdBatchInput):
    try:
        svc = FinStdService(
            provider=input.embeddingOptions.provider,
            model=input.embeddingOptions.model,
            index_path=input.embeddingOptions.indexPath,
            meta_path=input.embeddingOptions.metaPath,
        )
        out = []
        for t in input.texts:
            out.append({"query": t, "results": svc.search(t, limit=input.topK)})
        return {"items": out}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
