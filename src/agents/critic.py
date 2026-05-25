"""
Critic / Verifier Agent.

Evaluates whether the generated answer is:
  - Grounded in the retrieved context
  - Directly answering the user question
  - Complete (no major missing facts)
  - Free of unsupported claims / hallucination risk

Produces a structured decision:
  APPROVE           — answer is good; proceed to finaliser
  RETRIEVE_AGAIN    — answer is weak but query was fine; fetch more/different chunks
  REWRITE_QUERY     — query missed the mark; rewrite and retrieve again
  FAIL_GRACEFULLY   — evidence is fundamentally absent; surface a graceful fallback
"""

from __future__ import annotations

import json
import re
from typing import Tuple

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import settings
from src.graph.state import CriticDecision
from src.logger import get_logger

logger = get_logger(__name__)

# ── Pydantic output schema ─────────────────────────────────────────────────────

class CriticOutput(BaseModel):
    decision: CriticDecision = Field(
        description="One of: APPROVE, RETRIEVE_AGAIN, REWRITE_QUERY, FAIL_GRACEFULLY"
    )
    reason: str = Field(
        description="Short explanation (1–3 sentences) of why this decision was made."
    )
    grounding_score: float = Field(
        ge=0.0, le=1.0,
        description="0–1 score of how well the answer is grounded in the retrieved context."
    )
    completeness_score: float = Field(
        ge=0.0, le=1.0,
        description="0–1 score of how completely the answer addresses the question."
    )


# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a strict, impartial critic evaluating a RAG (Retrieval-Augmented Generation) answer.

Your task is to decide whether the generated answer should be accepted or if the pipeline should retry.

Evaluation criteria:
1. GROUNDING — Is every factual claim in the answer directly supported by the provided context chunks?
   If the answer asserts something not found in the context, grounding is low.
2. RELEVANCE — Does the answer actually address the user's question? Off-topic answers fail here.
3. COMPLETENESS — Does the answer cover the key aspects of the question given the available context?
4. HALLUCINATION RISK — Does the answer introduce specifics (numbers, names, dates, policies) that
   are NOT in the context? High hallucination risk means the answer should not be shown to users.
5. RETRIEVAL QUALITY — Does the context seem relevant to the question at all? If context is entirely
   off-topic, the query likely needs rewriting.

Decision rules:
- APPROVE: grounding ≥ 0.75 AND completeness ≥ 0.65 AND no apparent hallucinated facts.
- RETRIEVE_AGAIN: context is partially relevant but insufficient; the same/similar query
  might surface better chunks with a wider or differently-scored search.
- REWRITE_QUERY: context is largely irrelevant to the question; the query itself was poor.
- FAIL_GRACEFULLY: after multiple retries, or when the knowledge base clearly lacks the
  necessary information.

Respond ONLY with a valid JSON object matching this schema (no markdown fences):
{
  "decision": "APPROVE" | "RETRIEVE_AGAIN" | "REWRITE_QUERY" | "FAIL_GRACEFULLY",
  "reason": "<1-3 sentence explanation>",
  "grounding_score": <float 0.0–1.0>,
  "completeness_score": <float 0.0–1.0>
}
"""

HUMAN_TEMPLATE = """=== USER QUESTION ===
{question}

=== RETRIEVED CONTEXT CHUNKS ===
{context}

=== GENERATED ANSWER ===
{answer}

=== RETRY ATTEMPT ===
{retry_count} of {max_retries} maximum

Now evaluate and respond with the JSON decision object:"""


class CriticAgent:
    """Evaluates RAG answers and returns a structured decision."""

    def __init__(self) -> None:
        # Use a slightly higher temperature so the critic is not too lenient
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.1,
            openai_api_key=settings.openai_api_key,
        )

    def evaluate(
        self,
        question: str,
        context_docs: list,
        answer: str,
        retry_count: int,
        max_retries: int,
    ) -> Tuple[CriticDecision, str, CriticOutput]:
        """
        Evaluate the answer against the question and context.

        Returns:
            (decision, reason, full_output)
        """
        context_text = self._format_context(context_docs)

        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=HUMAN_TEMPLATE.format(
                    question=question,
                    context=context_text,
                    answer=answer,
                    retry_count=retry_count,
                    max_retries=max_retries,
                )
            ),
        ]

        try:
            response = self._llm.invoke(messages)
            output = self._parse_response(response.content)
            logger.info(
                "Critic decision=%s grounding=%.2f completeness=%.2f reason=%s",
                output.decision,
                output.grounding_score,
                output.completeness_score,
                output.reason,
            )
            return output.decision, output.reason, output

        except Exception as exc:
            logger.error("Critic evaluation failed: %s", exc)
            # On critic failure, approve rather than infinite loop
            fallback = CriticOutput(
                decision="APPROVE",
                reason=f"Critic encountered an error ({exc}); approving to avoid loop.",
                grounding_score=0.5,
                completeness_score=0.5,
            )
            return fallback.decision, fallback.reason, fallback

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _format_context(self, docs: list) -> str:
        if not docs:
            return "[NO CONTEXT RETRIEVED]"
        parts = []
        for i, doc in enumerate(docs, 1):
            chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")
            filename = doc.metadata.get("filename", "unknown")
            parts.append(
                f"[Chunk {i} | id={chunk_id} | file={filename}]\n{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)

    def _parse_response(self, raw: str) -> CriticOutput:
        """Parse raw LLM response into CriticOutput, with fallback."""
        # Strip markdown fences if present
        clean = re.sub(r"```json|```", "", raw).strip()
        try:
            data = json.loads(clean)
            return CriticOutput(**data)
        except (json.JSONDecodeError, Exception) as exc:
            logger.warning("Could not parse critic JSON (%s) — defaulting to RETRIEVE_AGAIN", exc)
            return CriticOutput(
                decision="RETRIEVE_AGAIN",
                reason="Critic response could not be parsed; attempting retrieval retry.",
                grounding_score=0.4,
                completeness_score=0.4,
            )
