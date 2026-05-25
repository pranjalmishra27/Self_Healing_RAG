"""
LangGraph Node Functions.

Each function receives the current RAGState, performs its task,
and returns a dict of state keys to update.
LangGraph merges the returned dict into the running state.
"""

from __future__ import annotations

from src.agents.critic import CriticAgent
from src.agents.query_rewriter import QueryRewriter
from src.chains.answer_chain import AnswerChain
from src.graph.state import RAGState, RetryRecord
from src.logger import get_logger
from src.retrieval.retriever import VectorRetriever

logger = get_logger(__name__)

# ── Singletons (initialised once per process) ──────────────────────────────────
_rewriter = QueryRewriter()
_retriever = VectorRetriever()
_answer_chain = AnswerChain()
_critic = CriticAgent()


# ── Node 1: Query Rewriter ─────────────────────────────────────────────────────

def node_query_rewriter(state: RAGState) -> dict:
    """
    Rewrites the original question into a better retrieval query.
    On retries it passes the critic's reason so the rewriter can correct course.
    """
    logger.info("[NODE] query_rewriter | retry=%d", state["retry_count"])

    critic_reason = state.get("critic_reason", "")
    rewritten = _rewriter.rewrite(
        question=state["original_question"],
        critic_reason=critic_reason,
    )
    return {"rewritten_question": rewritten}


# ── Node 2: Retriever ──────────────────────────────────────────────────────────

def node_retriever(state: RAGState) -> dict:
    """
    Retrieves the most relevant chunks from the vector store.
    Uses the rewritten question for the similarity query.
    """
    query = state.get("rewritten_question") or state["original_question"]
    logger.info("[NODE] retriever | query=%.80s", query)

    docs, sources = _retriever.retrieve(query=query)
    logger.info("[NODE] retriever | retrieved %d chunks", len(docs))
    return {
        "retrieved_documents": docs,
        "source_metadata": sources,
    }


# ── Node 3: Answer Generator ───────────────────────────────────────────────────

def node_answer_generator(state: RAGState) -> dict:
    """
    Generates a grounded answer using only the retrieved context chunks.
    """
    question = state.get("rewritten_question") or state["original_question"]
    docs = state.get("retrieved_documents", [])
    logger.info("[NODE] answer_generator | docs=%d", len(docs))

    answer = _answer_chain.generate(question=question, docs=docs)
    return {"generated_answer": answer}


# ── Node 4: Critic ─────────────────────────────────────────────────────────────

def node_critic(state: RAGState) -> dict:
    """
    Evaluates the generated answer for grounding and quality.
    Records a RetryRecord in retry_history for observability.
    """
    logger.info("[NODE] critic | retry=%d", state["retry_count"])

    decision, reason, _ = _critic.evaluate(
        question=state["original_question"],
        context_docs=state.get("retrieved_documents", []),
        answer=state.get("generated_answer", ""),
        retry_count=state["retry_count"],
        max_retries=state["max_retries"],
    )

    # Build retry record for traceability
    record = RetryRecord(
        retry_number=state["retry_count"],
        rewritten_question=state.get("rewritten_question", state["original_question"]),
        critic_decision=decision,
        critic_reason=reason,
        answer_preview=state.get("generated_answer", "")[:200],
    )
    history = list(state.get("retry_history", []))
    history.append(record)

    return {
        "critic_decision": decision,
        "critic_reason": reason,
        "retry_history": history,
    }


# ── Node 5: Finalizer ─────────────────────────────────────────────────────────

def node_finalizer(state: RAGState) -> dict:
    """
    Packages the approved answer with source citations.
    """
    logger.info("[NODE] finalizer")
    answer = state.get("generated_answer", "")
    sources = state.get("source_metadata", [])

    # Append source references if not already cited inline
    if sources and "(Source:" not in answer:
        source_list = "\n".join(
            f"  • {s['filename']}"
            + (f" (page {s['page']})" if s.get("page") else "")
            + (f" [score: {s['score']:.3f}]" if s.get("score") is not None else "")
            for s in sources
        )
        final = f"{answer}\n\n**Sources:**\n{source_list}"
    else:
        final = answer

    return {"final_response": final}


# ── Node 6: Fallback Response ──────────────────────────────────────────────────

def node_fallback_response(state: RAGState) -> dict:
    """
    Returns a graceful fallback when the pipeline cannot produce a confident answer.
    """
    logger.info("[NODE] fallback_response | retries exhausted or FAIL_GRACEFULLY")

    reason = state.get("critic_reason", "")
    msg = (
        "I was unable to find a well-grounded answer to your question in the indexed sources.\n\n"
        "This may mean:\n"
        "  • The topic is not covered in the ingested documents.\n"
        "  • The question requires information not yet indexed.\n"
        "  • The available evidence was insufficient after multiple retrieval attempts.\n\n"
    )
    if reason:
        msg += f"Last critic assessment: {reason}\n\n"
    msg += "Please try rephrasing your question or ingesting additional relevant documents."

    return {"final_response": msg}


# ── Routing function (used in conditional edges) ───────────────────────────────

def route_after_critic(state: RAGState) -> str:
    """
    Determines which node to visit after the critic runs.
    This is used as the `path` function in `add_conditional_edges`.

    Returns one of: "query_rewriter", "retriever", "finalizer", "fallback_response"
    """
    decision = state.get("critic_decision", "FAIL_GRACEFULLY")
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 3)

    # Always increment retry_count after a non-APPROVE decision
    if decision != "APPROVE":
        # Note: we can't mutate state here (read-only in routing fn)
        # The increment happens in the node that comes next via the bump_retry node
        pass

    if decision == "APPROVE":
        logger.info("[ROUTE] → finalizer")
        return "finalizer"

    if retry_count >= max_retries:
        logger.info("[ROUTE] → fallback_response (max retries reached)")
        return "fallback_response"

    if decision == "FAIL_GRACEFULLY":
        logger.info("[ROUTE] → fallback_response (FAIL_GRACEFULLY)")
        return "fallback_response"

    if decision == "REWRITE_QUERY":
        logger.info("[ROUTE] → query_rewriter")
        return "query_rewriter"

    # RETRIEVE_AGAIN
    logger.info("[ROUTE] → retriever")
    return "retriever"


def node_increment_retry(state: RAGState) -> dict:
    """
    Helper node that bumps the retry counter.
    Inserted between critic and the rewrite/retrieve branches.
    """
    new_count = state.get("retry_count", 0) + 1
    logger.info("[NODE] increment_retry → retry_count=%d", new_count)
    return {"retry_count": new_count}
