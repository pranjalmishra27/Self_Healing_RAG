"""
Pydantic request / response schemas for the FastAPI layer.
"""

from __future__ import annotations

from typing import List, Optional
from pydantic import BaseModel, Field


# ── /ingest ────────────────────────────────────────────────────────────────────

class IngestRequest(BaseModel):
    directory_path: str = Field(..., description="Path to the directory containing documents.")
    glob_pattern: str = Field("**/*.*", description="Glob pattern for file selection.")


class IngestResponse(BaseModel):
    status: str
    documents_ingested: int
    chunks_created: int
    message: str


# ── /ask ──────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, description="The user's question.")
    max_retries: int = Field(3, ge=0, le=10, description="Max critic retry loops.")
    top_k: Optional[int] = Field(None, ge=1, le=20, description="Number of chunks to retrieve.")


class SourceInfo(BaseModel):
    chunk_id: str
    filename: str
    page: Optional[int]
    source_type: str
    score: Optional[float]


class RetryEntry(BaseModel):
    retry_number: int
    rewritten_question: str
    critic_decision: str
    critic_reason: str
    answer_preview: str


class AskResponse(BaseModel):
    question: str
    final_answer: str
    sources: List[SourceInfo]
    critic_decision: str
    critic_reason: str
    retry_count: int
    retry_history: List[RetryEntry]
    rewritten_question: str


# ── /health ───────────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    vector_store_documents: int
    model: str
