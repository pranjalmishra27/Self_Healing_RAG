"""
LangGraph Workflow — Self-Healing RAG Pipeline.

Graph topology:
  START
    └→ query_rewriter
         └→ retriever
              └→ answer_generator
                   └→ critic
                        ├─(APPROVE)──────────────────────→ finalizer → END
                        ├─(RETRIEVE_AGAIN)──→ increment_retry → retriever
                        ├─(REWRITE_QUERY)───→ increment_retry → query_rewriter
                        └─(FAIL_GRACEFULLY / max_retries)→ fallback_response → END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from src.graph.nodes import (
    node_answer_generator,
    node_critic,
    node_fallback_response,
    node_finalizer,
    node_increment_retry,
    node_query_rewriter,
    node_retriever,
    route_after_critic,
)
from src.graph.state import RAGState
from src.logger import get_logger

logger = get_logger(__name__)

# ── Routing shim ───────────────────────────────────────────────────────────────
# LangGraph conditional edges need a function that returns a node *name* string.
# We split the routing from the increment so the counter is bumped in a separate node.

def _route_critic_to_branch(state: RAGState) -> str:
    """
    After critic runs decide which branch to take.
    APPROVE and FAIL_GRACEFULLY go directly to terminal nodes.
    RETRIEVE_AGAIN / REWRITE_QUERY go to the increment_retry node first.
    """
    from src.graph.nodes import route_after_critic
    decision = state.get("critic_decision", "FAIL_GRACEFULLY")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    if decision == "APPROVE":
        return "finalizer"
    if decision == "FAIL_GRACEFULLY" or retry_count >= max_retries:
        return "fallback_response"
    # RETRIEVE_AGAIN or REWRITE_QUERY → bump counter first
    return "increment_retry"


def _route_after_increment(state: RAGState) -> str:
    """After incrementing the counter, decide retriever or rewriter."""
    decision = state.get("critic_decision", "RETRIEVE_AGAIN")
    if decision == "REWRITE_QUERY":
        return "query_rewriter"
    return "retriever"


# ── Graph builder ──────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and compile the LangGraph state machine."""

    builder = StateGraph(RAGState)

    # Register nodes
    builder.add_node("query_rewriter", node_query_rewriter)
    builder.add_node("retriever", node_retriever)
    builder.add_node("answer_generator", node_answer_generator)
    builder.add_node("critic", node_critic)
    builder.add_node("increment_retry", node_increment_retry)
    builder.add_node("finalizer", node_finalizer)
    builder.add_node("fallback_response", node_fallback_response)

    # Linear flow from start
    builder.add_edge(START, "query_rewriter")
    builder.add_edge("query_rewriter", "retriever")
    builder.add_edge("retriever", "answer_generator")
    builder.add_edge("answer_generator", "critic")

    # Conditional branching from critic
    builder.add_conditional_edges(
        "critic",
        _route_critic_to_branch,
        {
            "finalizer": "finalizer",
            "fallback_response": "fallback_response",
            "increment_retry": "increment_retry",
        },
    )

    # After incrementing, branch to rewriter or retriever
    builder.add_conditional_edges(
        "increment_retry",
        _route_after_increment,
        {
            "query_rewriter": "query_rewriter",
            "retriever": "retriever",
        },
    )

    # Terminal edges
    builder.add_edge("finalizer", END)
    builder.add_edge("fallback_response", END)

    return builder.compile()


# ── Pipeline runner ────────────────────────────────────────────────────────────

class RAGPipeline:
    """High-level interface for running the self-healing RAG pipeline."""

    def __init__(self, max_retries: int | None = None) -> None:
        from src.config import settings
        self._max_retries = max_retries or settings.max_retries
        self._graph = build_graph()

    def run(self, question: str) -> dict:
        """
        Run the pipeline for a single question.

        Returns the final RAGState dict.
        """
        if not question or not question.strip():
            raise ValueError("Question must be a non-empty string.")

        initial_state: RAGState = {
            "original_question": question.strip(),
            "rewritten_question": "",
            "retrieved_documents": [],
            "source_metadata": [],
            "generated_answer": "",
            "critic_decision": "RETRIEVE_AGAIN",
            "critic_reason": "",
            "retry_count": 0,
            "max_retries": self._max_retries,
            "final_response": "",
            "retry_history": [],
        }

        logger.info("=== Pipeline START | question=%.120s ===", question)
        final_state = self._graph.invoke(initial_state)
        logger.info("=== Pipeline END | decision=%s ===", final_state.get("critic_decision"))
        return final_state
