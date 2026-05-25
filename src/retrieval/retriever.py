"""
Retrieval layer — wraps Chroma similarity search with metadata extraction.
"""

from __future__ import annotations

from typing import List, Tuple

from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

from src.config import settings
from src.graph.state import SourceMetadata
from src.logger import get_logger

logger = get_logger(__name__)


class VectorRetriever:
    """Thin retrieval wrapper around a Chroma collection."""

    def __init__(self) -> None:
        self._embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self._store: Chroma | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrieve(
        self,
        query: str,
        top_k: int | None = None,
        filter_metadata: dict | None = None,
    ) -> Tuple[List[Document], List[SourceMetadata]]:
        """
        Retrieve top-k chunks most similar to `query`.

        Returns:
            docs: list of LangChain Document objects
            sources: list of SourceMetadata dicts for citations
        """
        k = top_k or settings.top_k_retrieval
        store = self._get_store()

        try:
            # similarity_search_with_score returns (doc, score) pairs
            results: List[Tuple[Document, float]] = (
                store.similarity_search_with_score(query, k=k, filter=filter_metadata)
            )
        except Exception as exc:
            logger.error("Retrieval failed: %s", exc)
            return [], []

        if not results:
            logger.warning("No results found for query: %.80s", query)
            return [], []

        docs, sources = [], []
        for doc, score in results:
            docs.append(doc)
            sources.append(
                SourceMetadata(
                    chunk_id=doc.metadata.get("chunk_id", "unknown"),
                    filename=doc.metadata.get("filename", "unknown"),
                    page=doc.metadata.get("page"),
                    source_type=doc.metadata.get("source_type", "unknown"),
                    score=round(float(score), 4),
                )
            )
            logger.debug(
                "Retrieved chunk=%s score=%.4f",
                doc.metadata.get("chunk_id"),
                score,
            )

        return docs, sources

    def collection_size(self) -> int:
        try:
            return self._get_store()._collection.count()
        except Exception:
            return 0

    # ── Private helpers ────────────────────────────────────────────────────────

    def _get_store(self) -> Chroma:
        if self._store is None:
            self._store = Chroma(
                collection_name=settings.collection_name,
                embedding_function=self._embeddings,
                persist_directory=settings.vector_store_path,
            )
        return self._store
