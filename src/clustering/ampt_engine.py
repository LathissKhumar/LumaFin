"""AMPT Clustering Engine (HDBSCAN-based)

Discovers user-specific micro-categories from transaction embeddings.
"""
from __future__ import annotations

from typing import List, Dict, Any
from dataclasses import dataclass
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

try:
    import hdbscan  # type: ignore
except ImportError:  # Fallback placeholder
    hdbscan = None

from src.storage.database import SessionLocal
from src.utils.logger import get_logger
from src.embedder.encoder import TransactionEmbedder
try:
    from src.clustering.name_generator import generate_name  # optional
except Exception:
    generate_name = None  # type: ignore


@dataclass
class ClusterSummary:
    label: str
    centroid: np.ndarray
    num_points: int
    silhouette: float
    metadata: Dict[str, Any]


class AMPTClusteringEngine:
    def __init__(self, min_cluster_size: int = 5, min_samples: int = 3):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.embedder = TransactionEmbedder()
        self.log = get_logger(self.__class__.__name__)

    def fetch_user_transactions(self, user_id: int, limit: int = 1000) -> List[Dict[str, Any]]:
        db: Session = SessionLocal()
        try:
            result = db.execute(text("""
                SELECT id, merchant, amount, description
                FROM transactions
                WHERE user_id = :uid
                ORDER BY date DESC
                LIMIT :lim
            """), {"uid": user_id, "lim": limit})
            rows = []
            for r in result:
                rows.append({
                    "id": r[0],
                    "merchant": r[1],
                    "amount": float(r[2]) if r[2] is not None else None,
                    "description": r[3],
                })
            return rows
        finally:
            db.close()

    def cluster_user(self, user_id: int) -> List[ClusterSummary]:
        txns = self.fetch_user_transactions(user_id)
        if len(txns) < 50:
            self.log.info("Insufficient transactions for clustering", extra={"user_id": user_id, "count": len(txns)})
            return []

        embeddings = self.embedder.encode_batch(txns)  # shape (N, dim)
        # Normalize embeddings for cosine distance
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings_norm = embeddings / np.clip(norms, 1e-9, None)

        if hdbscan is None:
            self.log.warning("HDBSCAN not installed; skipping clustering", extra={"user_id": user_id})
            return []

        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean"  # using normalized vectors
        )
        labels = clusterer.fit_predict(embeddings_norm)

        clusters: Dict[int, List[int]] = {}
        for idx, label in enumerate(labels):
            if label == -1:
                continue  # noise
            clusters.setdefault(label, []).append(idx)

        summaries: List[ClusterSummary] = []
        for label, indices in clusters.items():
            points = embeddings_norm[indices]
            centroid = points.mean(axis=0)
            silhouette = 0.5  # Placeholder; proper computation requires pairwise distances
            metadata = self._derive_metadata([txns[i] for i in indices])
            name = generate_name(metadata) if generate_name else self._generate_name(metadata)
            summaries.append(ClusterSummary(
                label=name,
                centroid=centroid,
                num_points=len(indices),
                silhouette=silhouette,
                metadata=metadata
            ))
        # Persist clusters
        if summaries:
            try:
                self.persist_clusters(user_id, summaries)
            except Exception as e:
                self.log.error("Failed persisting clusters", extra={"user_id": user_id, "error": str(e)})
        return summaries

    def _derive_metadata(self, cluster_txns: List[Dict[str, Any]]) -> Dict[str, Any]:
        merchants = [t["merchant"].lower() for t in cluster_txns]
        amounts = [t["amount"] for t in cluster_txns if t["amount"] is not None]
        merchant_freq: Dict[str, int] = {}
        for m in merchants:
            merchant_freq[m] = merchant_freq.get(m, 0) + 1
        top_merchants = sorted(merchant_freq.items(), key=lambda x: x[1], reverse=True)[:3]
        avg_amount = float(np.mean(amounts)) if amounts else None
        return {
            "top_merchants": [m for m, _ in top_merchants],
            "avg_amount": avg_amount,
            "num_transactions": len(cluster_txns),
        }

    def _generate_name(self, metadata: Dict[str, Any]) -> str:
        tops = metadata.get("top_merchants", [])
        if tops:
            if len(tops) == 1:
                return f"{tops[0].title()} Pattern"
            else:
                return f"{tops[0].title()} & {tops[1].title()} Mix"
        return "Misc Cluster"

    def persist_clusters(self, user_id: int, clusters: List[ClusterSummary]) -> None:
        """Persist cluster centroids into personal_centroids table.

        Assumes schema:
          personal_centroids(user_id INT, category_name TEXT, centroid_vector JSONB, metadata JSONB,
                             num_transactions INT, updated_at TIMESTAMPTZ DEFAULT now(),
                             UNIQUE(user_id, category_name))
        The centroid_vector stored as list[float]. Adjust if pgvector is used.
        """
        db: Session = SessionLocal()
        try:
            for c in clusters:
                metadata = dict(c.metadata)
                metadata["silhouette"] = c.silhouette
                metadata["num_transactions"] = c.num_points
                centroid_list = c.centroid.tolist()
                db.execute(text("""
                    INSERT INTO personal_centroids (user_id, category_name, centroid_vector, metadata, num_transactions)
                    VALUES (:uid, :cat, :vec, :meta, :num)
                    ON CONFLICT (user_id, category_name)
                    DO UPDATE SET centroid_vector = EXCLUDED.centroid_vector,
                                  metadata = EXCLUDED.metadata,
                                  num_transactions = EXCLUDED.num_transactions,
                                  updated_at = now()
                """), {
                    "uid": user_id,
                    "cat": c.label,
                    "vec": centroid_list,
                    "meta": metadata,
                    "num": c.num_points,
                })
            db.commit()
            self.log.info("Persisted personal centroids", extra={"user_id": user_id, "count": len(clusters)})
        finally:
            db.close()


if __name__ == "__main__":
    engine = AMPTClusteringEngine()
    user_id = 1
    clusters = engine.cluster_user(user_id)
    log = get_logger("ampt_demo")
    log.info("Cluster generation complete", extra={"user_id": user_id, "count": len(clusters)})
    for c in clusters:
        log.info("Cluster summary", extra={"label": c.label, "size": c.num_points, "silhouette": round(c.silhouette, 4)})
