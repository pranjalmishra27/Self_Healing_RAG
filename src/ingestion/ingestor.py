"""
Document ingestion pipeline.
- Loads PDFs, text files, and Markdown from a directory.
- Splits into overlapping chunks.
- Embeds and stores in Chroma.
- Attaches rich metadata to every chunk.
"""

from __future__ import annotations

import hashlib
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List

from langchain_community.document_loaders import (
    DirectoryLoader,
    PyPDFLoader,
    TextLoader,
    UnstructuredMarkdownLoader,
)
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

from src.config import settings
from src.logger import get_logger

logger = get_logger(__name__)


def _chunk_id(doc: Document, index: int) -> str:
    """Deterministic chunk id based on source + content hash."""
    content_hash = hashlib.md5(doc.page_content.encode()).hexdigest()[:8]
    source = doc.metadata.get("source", "unknown")
    return f"{Path(source).stem}_{index}_{content_hash}"


class DocumentIngestor:
    """Load, split, embed, and store documents."""

    def __init__(self) -> None:
        self.embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            openai_api_key=settings.openai_api_key,
        )
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            add_start_index=True,
        )
        self._vector_store: Chroma | None = None

    # ── Public API ─────────────────────────────────────────────────────────────

    def ingest_directory(self, directory: str, glob: str = "**/*.*") -> dict:
        """
        Ingest all supported documents in a directory.
        Returns a summary dict with counts.
        """
        path = Path(directory)
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")

        raw_docs: List[Document] = []
        raw_docs.extend(self._load_pdfs(path))
        raw_docs.extend(self._load_text_files(path))
        raw_docs.extend(self._load_markdown_files(path))

        if not raw_docs:
            logger.warning("No documents found in %s", directory)
            return {"documents_ingested": 0, "chunks_created": 0}

        logger.info("Loaded %d raw documents from %s", len(raw_docs), directory)

        chunks = self._split_documents(raw_docs)
        logger.info("Created %d chunks", len(chunks))

        self._store_chunks(chunks)

        return {
            "documents_ingested": len(raw_docs),
            "chunks_created": len(chunks),
        }

    def ingest_text(self, text: str, metadata: dict | None = None) -> dict:
        """Ingest a raw text string directly (e.g. from API)."""
        doc = Document(
            page_content=text,
            metadata={
                "source": "direct_input",
                "source_type": "text",
                "ingested_at": datetime.now(timezone.utc).isoformat(),
                **(metadata or {}),
            },
        )
        chunks = self._split_documents([doc])
        self._store_chunks(chunks)
        return {"documents_ingested": 1, "chunks_created": len(chunks)}

    def get_vector_store(self) -> Chroma:
        """Return (or lazily open) the persisted Chroma collection."""
        if self._vector_store is None:
            self._vector_store = Chroma(
                collection_name=settings.collection_name,
                embedding_function=self.embeddings,
                persist_directory=settings.vector_store_path,
            )
        return self._vector_store

    def collection_size(self) -> int:
        try:
            return self.get_vector_store()._collection.count()
        except Exception:
            return 0

    # ── Private helpers ────────────────────────────────────────────────────────

    def _load_pdfs(self, path: Path) -> List[Document]:
        docs: List[Document] = []
        for pdf_path in path.rglob("*.pdf"):
            try:
                loader = PyPDFLoader(str(pdf_path))
                pages = loader.load()
                for page in pages:
                    page.metadata["source_type"] = "pdf"
                    page.metadata["filename"] = pdf_path.name
                docs.extend(pages)
            except Exception as exc:
                logger.error("Failed to load PDF %s: %s", pdf_path, exc)
        return docs

    def _load_text_files(self, path: Path) -> List[Document]:
        docs: List[Document] = []
        for txt_path in path.rglob("*.txt"):
            try:
                loader = TextLoader(str(txt_path), encoding="utf-8")
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source_type"] = "txt"
                    doc.metadata["filename"] = txt_path.name
                docs.extend(loaded)
            except Exception as exc:
                logger.error("Failed to load text file %s: %s", txt_path, exc)
        return docs

    def _load_markdown_files(self, path: Path) -> List[Document]:
        docs: List[Document] = []
        for md_path in path.rglob("*.md"):
            try:
                loader = TextLoader(str(md_path), encoding="utf-8")
                loaded = loader.load()
                for doc in loaded:
                    doc.metadata["source_type"] = "md"
                    doc.metadata["filename"] = md_path.name
                docs.extend(loaded)
            except Exception as exc:
                logger.error("Failed to load markdown %s: %s", md_path, exc)
        return docs

    def _split_documents(self, docs: List[Document]) -> List[Document]:
        chunks = self.splitter.split_documents(docs)
        now = datetime.now(timezone.utc).isoformat()
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_id"] = _chunk_id(chunk, i)
            chunk.metadata.setdefault("source_type", "unknown")
            chunk.metadata.setdefault("filename", "unknown")
            chunk.metadata.setdefault("page", None)
            chunk.metadata["ingested_at"] = now
        return chunks

    def _store_chunks(self, chunks: List[Document]) -> None:
        os.makedirs(settings.vector_store_path, exist_ok=True)
        vs = Chroma(
            collection_name=settings.collection_name,
            embedding_function=self.embeddings,
            persist_directory=settings.vector_store_path,
        )
        # Add in batches to avoid OpenAI rate limits
        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            vs.add_documents(batch)
            logger.debug("Stored batch %d-%d", i, i + len(batch))

        self._vector_store = vs
        logger.info("Vector store updated — total chunks stored")
