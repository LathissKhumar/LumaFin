"""Active learning selection utilities."""
from __future__ import annotations

from typing import List, Dict, Any
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.storage.database import SessionLocal


def select_uncertain(low: float = 0.45, high: float = 0.65, limit: int = 50) -> List[Dict[str, Any]]:
    """Select transactions with uncertain predictions for labeling.

    Assumes `transactions` table has columns: id, user_id, merchant, amount, description, predicted_category, confidence.
    """
    db: Session = SessionLocal()
    try:
        rows = db.execute(text(
            """
            SELECT id, user_id, merchant, amount, description, predicted_category, confidence
            FROM transactions
            WHERE confidence BETWEEN :low AND :high
            AND (is_corrected IS NULL OR is_corrected = FALSE)
            ORDER BY confidence ASC
            LIMIT :lim
            """
        ), {"low": low, "high": high, "lim": limit}).fetchall()
        out = []
        for r in rows:
            out.append({
                "id": r[0],
                "user_id": r[1],
                "merchant": r[2],
                "amount": float(r[3]) if r[3] is not None else 0.0,
                "description": r[4],
                "predicted_category": r[5],
                "confidence": float(r[6]) if r[6] is not None else 0.0,
            })
        return out
    finally:
        db.close()


__all__ = ["select_uncertain"]
