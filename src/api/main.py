"""
FastAPI application — exposes the Self-Healing RAG pipeline via HTTP.

Endpoints:
  GET  /health      — health + vector store info
  POST /ingest      — ingest documents from a directory
  POST /ask         — run the RAG pipeline for a question
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.api.schemas import (
    AskRequest,
    AskResponse,
    HealthResponse,
    IngestRequest,
    IngestResponse,
    RetryEntry,
    SourceInfo,
)
from src.config import settings
from src.graph.workflow import RAGPipeline
from src.ingestion.ingestor import DocumentIngestor
from src.logger import get_logger

logger = get_logger(__name__)

# ── Singletons ─────────────────────────────────────────────────────────────────
_ingestor: DocumentIngestor | None = None
_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    global _ingestor, _pipeline
    logger.info("Starting Self-Healing RAG API…")
    _ingestor = DocumentIngestor()
    _pipeline = RAGPipeline()
    yield
    logger.info("Shutting down.")


app = FastAPI(
    title="Self-Healing RAG Pipeline",
    description="Intelligent RAG with automatic critic-driven retry loops.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── /health ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    doc_count = _ingestor.collection_size() if _ingestor else 0
    return HealthResponse(
        status="healthy",
        vector_store_documents=doc_count,
        model=settings.llm_model,
    )


# ── /ingest ───────────────────────────────────────────────────────────────────

@app.post("/ingest", response_model=IngestResponse, tags=["Ingestion"])
def ingest(req: IngestRequest) -> IngestResponse:
    if _ingestor is None:
        raise HTTPException(status_code=503, detail="Ingestor not initialised.")
    try:
        result = _ingestor.ingest_directory(
            directory=req.directory_path,
        )
        return IngestResponse(
            status="success",
            documents_ingested=result["documents_ingested"],
            chunks_created=result["chunks_created"],
            message=f"Ingestion complete from {req.directory_path}",
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ingestion error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── /ask ──────────────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse, tags=["RAG Pipeline"])
def ask(req: AskRequest) -> AskResponse:
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialised.")

    # Override max_retries per-request if provided
    pipeline = RAGPipeline(max_retries=req.max_retries)

    t0 = time.perf_counter()
    try:
        state = pipeline.run(req.question)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Pipeline error")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    elapsed = time.perf_counter() - t0
    logger.info("Pipeline completed in %.2fs", elapsed)

    sources = [
        SourceInfo(
            chunk_id=s["chunk_id"],
            filename=s["filename"],
            page=s.get("page"),
            source_type=s["source_type"],
            score=s.get("score"),
        )
        for s in state.get("source_metadata", [])
    ]

    history = [
        RetryEntry(
            retry_number=r["retry_number"],
            rewritten_question=r["rewritten_question"],
            critic_decision=r["critic_decision"],
            critic_reason=r["critic_reason"],
            answer_preview=r["answer_preview"],
        )
        for r in state.get("retry_history", [])
    ]

    return AskResponse(
        question=state["original_question"],
        final_answer=state.get("final_response", ""),
        sources=sources,
        critic_decision=state.get("critic_decision", ""),
        critic_reason=state.get("critic_reason", ""),
        retry_count=state.get("retry_count", 0),
        retry_history=history,
        rewritten_question=state.get("rewritten_question", req.question),
    )
