"""Nightly incremental tasks: refresh FAISS, retrain reranker (placeholder)."""
from __future__ import annotations

import os
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.storage.database import SessionLocal
from src.indexer.faiss_builder import FAISSIndexBuilder
from src.embedder.encoder import TransactionEmbedder
from src.utils.logger import get_logger


log = get_logger("incremental")


def refresh_faiss_index() -> int:
    """Rebuild FAISS index from global_examples table and save to disk."""
    db: Session = SessionLocal()
    embedder = TransactionEmbedder()
    try:
        rows = db.execute(text(
            """
            SELECT ge.id, ge.merchant, ge.amount, ge.description, 
                   gt.category_name, ge.embedding
            FROM global_examples ge
            JOIN global_taxonomy gt ON ge.category_id = gt.id
            ORDER BY ge.id
            """
        )).fetchall()
        if not rows:
            log.warning("No global examples found; index not rebuilt")
            return 0
        
        # Convert stored embeddings back to numpy arrays
        examples = []
        embeddings = []
        for row in rows:
            example = {
                'id': row[0],
                'merchant': row[1],
                'amount': float(row[2]) if row[2] else 0.0,
                'description': row[3],
                'category': row[4],
            }
            # Handle pgvector embedding format
            if isinstance(row[5], list):
                embedding = np.array(row[5], dtype=np.float32)
            else:
                embedding = np.array(row[5])
            examples.append(example)
            embeddings.append(embedding)
        
        embeddings = np.array(embeddings, dtype=np.float32)
        builder = FAISSIndexBuilder(embedder.dimension)
        builder.build_index(embeddings, examples)
        builder.save()
        log.info("FAISS index refreshed", extra={"count": len(rows)})
        return len(rows)
    finally:
        db.close()


def rebuild_faiss_index() -> int:
    """Alias for refresh_faiss_index for Celery compatibility."""
    return refresh_faiss_index()


if __name__ == "__main__":
    refresh_faiss_index()
