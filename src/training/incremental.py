"""Nightly incremental tasks: refresh FAISS, retrain reranker (placeholder)."""
from __future__ import annotations

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
            SELECT ge.example_id, gt.category_name, ge.text
            FROM global_examples ge
            JOIN global_taxonomy gt ON ge.category_id = gt.category_id
            ORDER BY ge.example_id
            """
        )).fetchall()
        if not rows:
            log.warning("No global examples found; index not rebuilt")
            return 0
        examples = [{"merchant": r[2], "amount": 0.0, "description": None} for r in rows]
        embeddings = embedder.encode_batch(examples)
        meta = [{"id": r[0], "category": r[1], "merchant": r[2], "amount": 0.0} for r in rows]
        builder = FAISSIndexBuilder(embedder.dimension)
        builder.build_index(embeddings, meta)
        builder.save()
        log.info("FAISS index refreshed", extra={"count": len(rows)})
        return len(rows)
    finally:
        db.close()


if __name__ == "__main__":
    refresh_faiss_index()
