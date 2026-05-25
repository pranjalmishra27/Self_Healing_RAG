"""
Query Rewriter Agent.
Rewrites the user's question into a more precise retrieval query.
On subsequent retries it also receives the critic's reason so it can
make a smarter reformulation.
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

# ── Prompt ─────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a search query optimisation expert.
Your job is to rewrite a user's question into an optimal retrieval query
for a semantic vector search over a document knowledge base.

Rules:
1. Extract the core information need — remove conversational filler.
2. Expand acronyms and abbreviations.
3. Add relevant synonyms or related terms that might appear in source documents.
4. Keep the rewritten query concise (≤ 20 words).
5. If a previous retrieval failed and a critic reason is provided, incorporate
   that feedback to broaden or sharpen the query.
6. Return ONLY the rewritten query — no explanations, no bullet points, no quotes.
"""

HUMAN_TEMPLATE = """Original question: {question}

Previous critic feedback (empty if first attempt): {critic_reason}

Rewritten retrieval query:"""


class QueryRewriter:
    """Rewrites user questions into better retrieval queries."""

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.0,
            openai_api_key=settings.openai_api_key,
        )

    def rewrite(self, question: str, critic_reason: str = "") -> str:
        """
        Rewrite `question` into an optimised retrieval query.

        Args:
            question: original user question
            critic_reason: feedback from the critic on the previous attempt
        Returns:
            Rewritten query string
        """
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=HUMAN_TEMPLATE.format(
                    question=question,
                    critic_reason=critic_reason or "N/A",
                )
            ),
        ]
        try:
            response = self._llm.invoke(messages)
            rewritten = response.content.strip().strip('"').strip("'")
            logger.info("Rewritten query: %s", rewritten)
            return rewritten
        except Exception as exc:
            logger.error("Query rewriting failed: %s — using original", exc)
            return question
