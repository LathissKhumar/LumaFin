"""Personal centroid matcher.

Given a transaction embedding and user ID, return best matching personal category.
"""
from __future__ import annotations

from typing import Optional, Dict, Any
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.storage.database import SessionLocal

SIMILARITY_THRESHOLD = 0.80
MIN_SUPPORT = 5


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-9)
    b_norm = b / (np.linalg.norm(b) + 1e-9)
    return float(np.dot(a_norm, b_norm))


def fetch_centroids(user_id: int) -> list[dict[str, Any]]:
    db: Session = SessionLocal()
    try:
        result = db.execute(text("""
            SELECT id, category_name, centroid_vector, metadata
            FROM personal_centroids
            WHERE user_id = :uid
        """), {"uid": user_id})
        centroids = []
        for r in result:
            vector = np.array(r[2], dtype=np.float32) if isinstance(r[2], list) else np.array(r[2])
            centroids.append({
                "id": r[0],
                "category_name": r[1],
                "vector": vector,
                "metadata": r[3],
            })
        return centroids
    finally:
        db.close()


def match_personal_category(user_id: int, embedding: np.ndarray, label: str | None = None) -> Optional[Dict[str, Any]]:
    centroids = fetch_centroids(user_id)
    if not centroids:
        return None

    best = None
    best_sim = -1.0
    for c in centroids:
        sim = cosine(embedding, c["vector"])
        # If centroid metadata contains a dominant_label and user label matches, boost similarity
        dom_label = (c.get("metadata") or {}).get("dominant_label")
        if label and dom_label and label.lower().strip() == dom_label.lower().strip():
            sim = min(1.0, sim + 0.05)  # small boost
        support = c["metadata"].get("num_transactions", 0) if c.get("metadata") else 0
        if sim > best_sim and sim >= SIMILARITY_THRESHOLD and support >= MIN_SUPPORT:
            best = c
            best_sim = sim
    if best:
        return {"category": best["category_name"], "similarity": best_sim, "metadata": best.get("metadata")}
    return None
