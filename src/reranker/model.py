"""Reranker module.

Provides a lightweight reranker abstraction. If cross-encoder / xgboost are
available, they can be plugged in. By default, falls back to a heuristic using
retrieval similarities and simple votes.
"""
from __future__ import annotations

from typing import List, Dict, Any, Tuple

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification  # type: ignore
    _HAS_CE = True
except Exception:
    _HAS_CE = False

try:
    import xgboost as xgb  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False


class Reranker:
    def __init__(self):
        self.has_ce = _HAS_CE
        self.has_xgb = _HAS_XGB
        self.ce_model = None
        self.ce_tokenizer = None
        self.xgb_model = None
        if self.has_ce:
            try:
                self.ce_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
                self.ce_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception:
                # Fallback to heuristic if model can't be loaded (e.g., offline)
                self.has_ce = False

    def _heuristic_score(self, candidates: List[Dict[str, Any]]) -> Tuple[str, float]:
        # Aggregate by category using max similarity
        by_cat: Dict[str, float] = {}
        for c in candidates:
            cat = c.get("category") or c.get("category_name") or "Unknown"
            sim = float(c.get("similarity", 0.0))
            by_cat[cat] = max(by_cat.get(cat, 0.0), sim)
        if not by_cat:
            return "Uncategorized", 0.0
        best_cat = max(by_cat.items(), key=lambda kv: kv[1])
        # Map similarity to a soft confidence (0.4-0.8)
        conf = 0.4 + 0.4 * max(0.0, min(1.0, best_cat[1]))
        return best_cat[0], conf

    def rerank(self, query_text: str, candidates: List[Dict[str, Any]]) -> Tuple[str, float, List[Dict[str, Any]]]:
        """Return (category, confidence, enriched_candidates).

        candidates item schema expected keys: merchant, amount, category, similarity
        """
        if not candidates:
            return "Uncategorized", 0.0, []

        # Heuristic fallback (no CE/XGB)
        cat, conf = self._heuristic_score(candidates)
        return cat, conf, candidates


_global_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _global_reranker
    if _global_reranker is None:
        _global_reranker = Reranker()
    return _global_reranker


__all__ = ["Reranker", "get_reranker"]
