"""
Answer Generation Chain.

Generates a grounded, cited answer from retrieved context chunks.
The prompt strongly instructs the model to use ONLY the provided context
and to say "I don't know" when evidence is insufficient.
"""

from __future__ import annotations

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)

# ── Prompts ────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are a precise, fact-grounded question-answering assistant.

STRICT RULES — follow these exactly:
1. Answer ONLY from the context chunks provided below.
2. Do NOT use any external knowledge, prior training data, or assumptions.
3. If the context does not contain enough information to answer the question,
   respond with exactly: "Based on the available indexed sources, I do not have
   enough information to answer this question."
4. Every specific claim must be traceable to a context chunk.
5. When referencing a fact, mention its source in parentheses, e.g. (Source: filename.pdf).
6. Keep the answer clear, concise, and structured.
7. Do not pad the answer with unnecessary caveats or filler.
"""

HUMAN_TEMPLATE = """=== CONTEXT CHUNKS ===
{context}

=== USER QUESTION ===
{question}

Answer (grounded in the above context only):"""


class AnswerChain:
    """Generates answers strictly grounded in retrieved context."""

    def __init__(self) -> None:
        self._llm = ChatOpenAI(
            model=settings.llm_model,
            temperature=0.0,          # deterministic generation
            openai_api_key=settings.openai_api_key,
        )

    def generate(self, question: str, docs: list[Document]) -> str:
        """
        Generate an answer for `question` using `docs` as the sole context.

        Args:
            question: user question (may be rewritten)
            docs: retrieved Document objects
        Returns:
            Generated answer string
        """
        if not docs:
            logger.warning("No documents provided to answer generator — returning fallback")
            return (
                "Based on the available indexed sources, I do not have enough "
                "information to answer this question."
            )

        context = self._format_context(docs)
        messages = [
            SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(
                content=HUMAN_TEMPLATE.format(context=context, question=question)
            ),
        ]

        try:
            response = self._llm.invoke(messages)
            answer = response.content.strip()
            logger.info("Generated answer (%d chars)", len(answer))
            return answer
        except Exception as exc:
            logger.error("Answer generation failed: %s", exc)
            return (
                "An error occurred while generating the answer. "
                "Please try again."
            )

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _format_context(self, docs: list[Document]) -> str:
        parts = []
        for i, doc in enumerate(docs, 1):
            chunk_id = doc.metadata.get("chunk_id", f"chunk_{i}")
            filename = doc.metadata.get("filename", "unknown")
            page = doc.metadata.get("page")
            page_str = f" | page {page}" if page else ""
            parts.append(
                f"[Chunk {i} | id={chunk_id} | source={filename}{page_str}]\n"
                f"{doc.page_content}"
            )
        return "\n\n---\n\n".join(parts)
