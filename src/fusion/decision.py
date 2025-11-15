"""Prediction fusion layer implementing hierarchical decision pipeline.

Order:
1) Rules
2) Personal Centroids
3) Retrieval + Reranker
4) Fallback
"""
from __future__ import annotations

from typing import Dict, Any, Tuple, List

from src.embedder.encoder import TransactionEmbedder
from src.rules.engine import rule_engine
from src.retrieval.service import get_retrieval_service
from src.reranker.model import get_reranker
from src.clustering.centroid_matcher import match_personal_category, SIMILARITY_THRESHOLD
from src.utils.logger import get_logger


log = get_logger("fusion")
_embedder = TransactionEmbedder()
_retrieval = get_retrieval_service()
_reranker = get_reranker()


def _scale_centroid_conf(sim: float, base: float = 0.85, max_c: float = 0.95) -> float:
    return max(base, min(max_c, base + (min(1.0, sim) - SIMILARITY_THRESHOLD) * (max_c - base) / (1.0 - SIMILARITY_THRESHOLD)))


def decide(txn: Dict[str, Any]) -> Tuple[str, float, Dict[str, Any]]:
    """Return (category, confidence, explanation_dict).

    txn keys: merchant, amount, description(optional), user_id(optional)
    """
    merchant = txn["merchant"]
    amount = float(txn["amount"]) if txn.get("amount") is not None else 0.0
    description = txn.get("description")
    user_id = txn.get("user_id")

    # 1) Rules
    rule_result = rule_engine.apply_rules(merchant, amount)
    if rule_result:
        log.info("Rule match", extra={"rule": rule_result.name})
        return rule_result.name, 1.0, {"decision_path": "rule", "rule_matched": rule_result.name}

    # Embed once
    emb = _embedder.encode_transaction(merchant=merchant, amount=amount, description=description)

    # 2) Personal centroids
    if user_id is not None:
        personal = match_personal_category(user_id, emb)
        if personal:
            conf = _scale_centroid_conf(personal["similarity"]) 
            log.info("Centroid match", extra={"user_id": user_id, "category": personal["category"], "sim": round(personal["similarity"], 4)})
            return personal["category"], conf, {"decision_path": "centroid", "centroid_similarity": personal["similarity"]}

    # 3) Retrieval + reranker
    try:
        results = _retrieval.retrieve_by_embedding(emb, k=20)
        if results:
            cat, conf, enriched = _reranker.rerank(merchant, results)
            return cat, conf, {"decision_path": "retrieval", "nearest_examples": [
                {
                    "merchant": r.get("merchant"),
                    "amount": r.get("amount"),
                    "category": r.get("category"),
                    "similarity": r.get("similarity"),
                } for r in enriched[:3]
            ]}
    except Exception as e:
        log.error("Retrieval/rerank failed", extra={"error": str(e)})

    # 4) Fallback
    return "Uncategorized", 0.0, {"decision_path": "fallback"}


__all__ = ["decide"]
