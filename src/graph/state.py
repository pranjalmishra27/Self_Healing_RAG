"""
RAGState — the single shared state object passed between all LangGraph nodes.
Every node reads from and writes to this TypedDict.
"""

from __future__ import annotations

from typing import Any, Literal, Optional
from typing_extensions import TypedDict


CriticDecision = Literal["APPROVE", "RETRIEVE_AGAIN", "REWRITE_QUERY", "FAIL_GRACEFULLY"]


class RetryRecord(TypedDict):
    """A snapshot of one retry attempt for observability / tracing."""
    retry_number: int
    rewritten_question: str
    critic_decision: CriticDecision
    critic_reason: str
    answer_preview: str  # first 200 chars of generated answer


class SourceMetadata(TypedDict):
    """Metadata attached to each retrieved chunk / citation."""
    chunk_id: str
    filename: str
    page: Optional[int]
    source_type: str          # e.g. "pdf", "txt", "web"
    score: Optional[float]    # cosine similarity score if available


class RAGState(TypedDict):
    """
    Full pipeline state.  Nodes read/write specific keys; LangGraph merges changes.
    """
    # ── Input ──────────────────────────────────────────────────────────────────
    original_question: str
    rewritten_question: str           # may equal original_question on first pass

    # ── Retrieval ──────────────────────────────────────────────────────────────
    retrieved_documents: list[Any]    # list[langchain_core.documents.Document]
    source_metadata: list[SourceMetadata]

    # ── Generation ─────────────────────────────────────────────────────────────
    generated_answer: str

    # ── Critic ─────────────────────────────────────────────────────────────────
    critic_decision: CriticDecision
    critic_reason: str

    # ── Loop control ───────────────────────────────────────────────────────────
    retry_count: int
    max_retries: int

    # ── Output ─────────────────────────────────────────────────────────────────
    final_response: str
    retry_history: list[RetryRecord]
