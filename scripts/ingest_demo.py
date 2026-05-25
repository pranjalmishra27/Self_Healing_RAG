#!/usr/bin/env python3
"""
One-shot script to ingest the sample documents.
Run: python scripts/ingest_demo.py
"""

import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.ingestion.ingestor import DocumentIngestor
from src.logger import get_logger

logger = get_logger("ingest_demo")


def main() -> None:
    sample_dir = Path(__file__).parent.parent / "data" / "sample_docs"
    if not sample_dir.exists():
        logger.error("Sample documents directory not found: %s", sample_dir)
        sys.exit(1)

    logger.info("Starting ingestion from: %s", sample_dir)
    ingestor = DocumentIngestor()

    result = ingestor.ingest_directory(str(sample_dir))

    logger.info("✅  Ingestion complete!")
    logger.info("   Documents loaded : %d", result["documents_ingested"])
    logger.info("   Chunks stored    : %d", result["chunks_created"])
    logger.info("   Vector store size: %d", ingestor.collection_size())


if __name__ == "__main__":
    main()
