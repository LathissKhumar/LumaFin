"""
Retrieval Service for Transaction Categorization

Queries FAISS index and returns top-k candidates with categories.
"""
from __future__ import annotations

from typing import List, Dict, Optional

import numpy as np

from src.indexer.faiss_builder import get_faiss_index, FAISSIndexBuilder
from src.embedder.encoder import TransactionEmbedder


class RetrievalService:
    """Query FAISS index for similar transactions."""

    def __init__(
        self,
        embedder: TransactionEmbedder | None = None,
        faiss_index: FAISSIndexBuilder | None = None
    ):
        self.embedder = embedder or TransactionEmbedder()
        self.faiss_index = faiss_index or get_faiss_index()

    def retrieve(
        self,
        merchant: str,
        amount: float | None = None,
        description: str | None = None,
        k: int = 20
    ) -> List[Dict]:
        """
        Retrieve top-k similar transactions.

        Args:
            merchant: Merchant name
            amount: Transaction amount
            description: Optional description
            k: Number of results to return

        Returns:
            List of dicts with keys: id, category, merchant, amount, similarity
        """
        # Generate embedding
        embedding = self.embedder.encode_transaction(
            merchant=merchant,
            amount=amount,
            description=description
        )

        # Query FAISS
        results = self.faiss_index.search(embedding, k=k)

        return results

    def retrieve_by_embedding(self, embedding: np.ndarray, k: int = 20) -> List[Dict]:
        """Retrieve using pre-computed embedding."""
        return self.faiss_index.search(embedding, k=k)

    def get_category_votes(self, results: List[Dict]) -> Dict[str, float]:
        """
        Aggregate category votes from retrieval results.

        Returns dict mapping category -> weighted vote (sum of similarities).
        """
        votes: Dict[str, float] = {}
        for result in results:
            category = result['category']
            similarity = result['similarity']
            votes[category] = votes.get(category, 0.0) + similarity

        return votes

    def predict_from_votes(
        self,
        votes: Dict[str, float],
        threshold: float = 0.5
    ) -> tuple[str, float]:
        """
        Predict category from votes.

        Args:
            votes: Dict of category -> score
            threshold: Minimum confidence threshold

        Returns:
            (category, confidence) tuple
        """
        if not votes:
            return ("Uncategorized", 0.0)

        # Get top category
        sorted_votes = sorted(votes.items(), key=lambda x: x[1], reverse=True)
        top_category, top_score = sorted_votes[0]

        # Calculate confidence (normalized by total votes)
        total_votes = sum(votes.values())
        confidence = top_score / total_votes if total_votes > 0 else 0.0

        # Check threshold
        if confidence < threshold:
            return ("Uncategorized", confidence)

        return (top_category, confidence)


# Global retrieval service instance
_retrieval_service: Optional[RetrievalService] = None


def get_retrieval_service() -> RetrievalService:
    """Get or create global retrieval service instance."""
    global _retrieval_service
    if _retrieval_service is None:
        _retrieval_service = RetrievalService()
    return _retrieval_service
