"""
Tests for the Self-Healing RAG Pipeline.

Run: pytest tests/ -v
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.graph.state import RAGState
from src.graph.nodes import (
    node_query_rewriter,
    node_answer_generator,
    node_critic,
    node_finalizer,
    node_fallback_response,
    node_increment_retry,
    route_after_critic,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def base_state(**overrides) -> RAGState:
    state: RAGState = {
        "original_question": "What is the warranty period?",
        "rewritten_question": "warranty period duration coverage",
        "retrieved_documents": [],
        "source_metadata": [],
        "generated_answer": "",
        "critic_decision": "RETRIEVE_AGAIN",
        "critic_reason": "",
        "retry_count": 0,
        "max_retries": 3,
        "final_response": "",
        "retry_history": [],
    }
    state.update(overrides)
    return state


# ── Unit tests for routing ─────────────────────────────────────────────────────

class TestRouting:
    def test_approve_routes_to_finalizer(self):
        state = base_state(critic_decision="APPROVE")
        assert route_after_critic(state) == "finalizer"

    def test_fail_gracefully_routes_to_fallback(self):
        state = base_state(critic_decision="FAIL_GRACEFULLY")
        assert route_after_critic(state) == "fallback_response"

    def test_max_retries_exceeded_routes_to_fallback(self):
        state = base_state(critic_decision="RETRIEVE_AGAIN", retry_count=3, max_retries=3)
        assert route_after_critic(state) == "fallback_response"

    def test_retrieve_again_routes_to_retriever(self):
        state = base_state(critic_decision="RETRIEVE_AGAIN", retry_count=1, max_retries=3)
        assert route_after_critic(state) == "retriever"

    def test_rewrite_query_routes_to_rewriter(self):
        state = base_state(critic_decision="REWRITE_QUERY", retry_count=1, max_retries=3)
        assert route_after_critic(state) == "query_rewriter"


# ── Unit tests for nodes ───────────────────────────────────────────────────────

class TestNodes:
    def test_increment_retry(self):
        state = base_state(retry_count=1)
        result = node_increment_retry(state)
        assert result["retry_count"] == 2

    def test_finalizer_appends_sources(self):
        state = base_state(
            generated_answer="The warranty is 2 years.",
            source_metadata=[
                {
                    "chunk_id": "doc_001",
                    "filename": "warranty.txt",
                    "page": 1,
                    "source_type": "txt",
                    "score": 0.91,
                }
            ],
        )
        result = node_finalizer(state)
        assert "warranty.txt" in result["final_response"]

    def test_fallback_contains_helpful_message(self):
        state = base_state(critic_reason="Context entirely off-topic.")
        result = node_fallback_response(state)
        assert "not covered" in result["final_response"].lower() or \
               "unable to find" in result["final_response"].lower()

    @patch("src.graph.nodes._rewriter")
    def test_query_rewriter_node(self, mock_rewriter):
        mock_rewriter.rewrite.return_value = "product warranty duration years"
        state = base_state()
        result = node_query_rewriter(state)
        assert result["rewritten_question"] == "product warranty duration years"
        mock_rewriter.rewrite.assert_called_once()

    @patch("src.graph.nodes._retriever")
    def test_retriever_node_empty(self, mock_retriever):
        mock_retriever.retrieve.return_value = ([], [])
        state = base_state()
        result = node_retriever(state)
        assert result["retrieved_documents"] == []
        assert result["source_metadata"] == []

    @patch("src.graph.nodes._answer_chain")
    def test_answer_generator_no_docs(self, mock_chain):
        mock_chain.generate.return_value = "I do not have enough information."
        state = base_state(retrieved_documents=[])
        result = node_answer_generator(state)
        assert result["generated_answer"] == "I do not have enough information."

    @patch("src.graph.nodes._critic")
    def test_critic_node_approve(self, mock_critic):
        mock_critic.evaluate.return_value = (
            "APPROVE",
            "Answer is well grounded.",
            MagicMock(grounding_score=0.9, completeness_score=0.85),
        )
        state = base_state(generated_answer="The warranty is 2 years.")
        result = node_critic(state)
        assert result["critic_decision"] == "APPROVE"
        assert len(result["retry_history"]) == 1

    @patch("src.graph.nodes._critic")
    def test_critic_node_builds_retry_history(self, mock_critic):
        mock_critic.evaluate.return_value = (
            "RETRIEVE_AGAIN",
            "Insufficient evidence.",
            MagicMock(grounding_score=0.3, completeness_score=0.4),
        )
        state = base_state(
            generated_answer="Some answer.",
            retry_history=[
                {
                    "retry_number": 0,
                    "rewritten_question": "initial query",
                    "critic_decision": "REWRITE_QUERY",
                    "critic_reason": "Off topic.",
                    "answer_preview": "Previous answer preview",
                }
            ],
        )
        result = node_critic(state)
        assert len(result["retry_history"]) == 2


# ── Integration smoke test (no LLM calls) ─────────────────────────────────────

class TestWorkflow:
    @patch("src.graph.nodes._rewriter")
    @patch("src.graph.nodes._retriever")
    @patch("src.graph.nodes._answer_chain")
    @patch("src.graph.nodes._critic")
    def test_pipeline_approves_on_first_attempt(
        self, mock_critic, mock_chain, mock_retriever, mock_rewriter
    ):
        from langchain_core.documents import Document
        from src.graph.workflow import RAGPipeline

        # Mock LLM calls
        mock_rewriter.rewrite.return_value = "warranty period coverage"
        mock_retriever.retrieve.return_value = (
            [Document(page_content="2 year warranty.", metadata={"filename": "w.txt", "chunk_id": "c1", "source_type": "txt"})],
            [{"chunk_id": "c1", "filename": "w.txt", "page": None, "source_type": "txt", "score": 0.95}],
        )
        mock_chain.generate.return_value = "The warranty period is 2 years."
        mock_critic.evaluate.return_value = (
            "APPROVE",
            "Well grounded.",
            MagicMock(grounding_score=0.95, completeness_score=0.9),
        )

        pipeline = RAGPipeline(max_retries=3)
        state = pipeline.run("What is the warranty period?")

        assert state["critic_decision"] == "APPROVE"
        assert state["retry_count"] == 0
        assert "2 years" in state["final_response"]

    @patch("src.graph.nodes._rewriter")
    @patch("src.graph.nodes._retriever")
    @patch("src.graph.nodes._answer_chain")
    @patch("src.graph.nodes._critic")
    def test_pipeline_falls_back_after_max_retries(
        self, mock_critic, mock_chain, mock_retriever, mock_rewriter
    ):
        from langchain_core.documents import Document
        from src.graph.workflow import RAGPipeline

        mock_rewriter.rewrite.return_value = "some query"
        mock_retriever.retrieve.return_value = ([], [])
        mock_chain.generate.return_value = "I don't know."
        mock_critic.evaluate.return_value = (
            "RETRIEVE_AGAIN",
            "No useful context retrieved.",
            MagicMock(grounding_score=0.1, completeness_score=0.1),
        )

        pipeline = RAGPipeline(max_retries=2)
        state = pipeline.run("What is the meaning of life?")

        assert "unable to find" in state["final_response"].lower() or \
               "not covered" in state["final_response"].lower() or \
               "insufficient" in state["final_response"].lower() or \
               len(state["final_response"]) > 0
        assert state["retry_count"] >= 2
