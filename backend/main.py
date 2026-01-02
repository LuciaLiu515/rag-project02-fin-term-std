from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Literal, Optional

from services.fin_std_service import FinStdService


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
