"""Celery worker for processing user feedback and updating personal centroids.

Reads from feedback_queue and applies:
- Update personal centroid via moving average
- Optionally append corrected example to global_examples (TODO)
"""
from __future__ import annotations

import os
from typing import Dict, Any
from celery import Celery
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.storage.database import SessionLocal
from src.embedder.encoder import TransactionEmbedder
from src.utils.logger import get_logger


REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")
celery_app = Celery("feedback_worker", broker=REDIS_URL, backend=REDIS_URL)
log = get_logger("feedback_worker")
embedder = TransactionEmbedder()


def _get_transaction(db: Session, transaction_id: int) -> Dict[str, Any] | None:
    res = db.execute(text(
        """
        SELECT id, user_id, merchant, amount, description
        FROM transactions
        WHERE id = :tid
        """
    ), {"tid": transaction_id})
    row = res.fetchone()
    if not row:
        return None
    return {
        "id": row[0],
        "user_id": row[1],
        "merchant": row[2],
        "amount": float(row[3]) if row[3] is not None else 0.0,
        "description": row[4],
    }


def _upsert_centroid(db: Session, user_id: int, category: str, emb: list[float]) -> None:
    # Moving average update: new = 0.9*old + 0.1*emb
    # If no row, insert
    existing = db.execute(text(
        """
        SELECT centroid_vector, num_transactions, metadata
        FROM personal_centroids
        WHERE user_id = :uid AND category_name = :cat
        """
    ), {"uid": user_id, "cat": category}).fetchone()

    if existing:
        old_vec, num_txn, metadata = existing
        if isinstance(old_vec, list):
            old = old_vec
        else:
            old = old_vec  # assume list-like
        new = [0.9 * ov + 0.1 * nv for ov, nv in zip(old, emb)]
        new_num = int(num_txn or 0) + 1
        meta = metadata or {}
        meta["num_transactions"] = new_num
        db.execute(text(
            """
            UPDATE personal_centroids
            SET centroid_vector = :vec, num_transactions = :num, metadata = :meta, updated_at = now()
            WHERE user_id = :uid AND category_name = :cat
            """
        ), {"vec": new, "num": new_num, "meta": meta, "uid": user_id, "cat": category})
    else:
        meta = {"num_transactions": 1}
        db.execute(text(
            """
            INSERT INTO personal_centroids (user_id, category_name, centroid_vector, metadata, num_transactions)
            VALUES (:uid, :cat, :vec, :meta, :num)
            """
        ), {"uid": user_id, "cat": category, "vec": emb, "meta": meta, "num": 1})


@celery_app.task(name="process_feedback_batch")
def process_feedback_batch(limit: int = 100) -> int:
    """Process up to `limit` feedback rows and update centroids."""
    db: Session = SessionLocal()
    processed = 0
    try:
        rows = db.execute(text(
            """
            SELECT feedback_id, user_id, transaction_id, correct_category
            FROM feedback_queue
            WHERE processed = FALSE
            ORDER BY created_at ASC
            LIMIT :lim
            """
        ), {"lim": limit}).fetchall()
        for fid, uid, tid, correct_cat in rows:
            txn = _get_transaction(db, tid)
            if not txn:
                log.warning("Transaction not found for feedback", extra={"feedback_id": fid, "transaction_id": tid})
                db.execute(text("UPDATE feedback_queue SET processed = TRUE WHERE feedback_id = :fid"), {"fid": fid})
                continue
            emb = embedder.encode_transaction(txn["merchant"], txn["amount"], txn["description"]).tolist()
            _upsert_centroid(db, uid, correct_cat, emb)
            db.execute(text("UPDATE feedback_queue SET processed = TRUE WHERE feedback_id = :fid"), {"fid": fid})
            processed += 1
        db.commit()
        log.info("Processed feedback batch", extra={"count": processed})
        return processed
    except Exception as e:
        db.rollback()
        log.error("Feedback batch failed", extra={"error": str(e)})
        return processed
    finally:
        db.close()


__all__ = ["celery_app", "process_feedback_batch"]
