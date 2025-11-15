"""Explanation builder for predictions.

Combines nearest examples, rule traces, cluster coherence, and optional SHAP
attributions into a single Explanation model.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

from src.models import Explanation


def build_explanation(
    decision_path: str,
    nearest_examples: Optional[List[Dict[str, Any]]] = None,
    rule_matched: Optional[str] = None,
    centroid_similarity: Optional[float] = None,
    shap_values: Optional[Dict[str, float]] = None,
) -> Explanation:
    return Explanation(
        decision_path=decision_path,
        nearest_examples=nearest_examples,
        rule_matched=rule_matched,
        centroid_similarity=centroid_similarity,
        feature_importance=shap_values,
    )


def summarize(expl: Explanation, category: str, confidence: float) -> str:
    base = f"Categorized as {category} with confidence {round(confidence * 100, 1)}%."
    if expl.decision_path == "rule" and expl.rule_matched:
        return base + f" Rule matched: {expl.rule_matched}."
    if expl.decision_path == "centroid" and expl.centroid_similarity is not None:
        return base + f" Personal pattern similarity: {round(expl.centroid_similarity * 100, 1)}%."
    if expl.decision_path == "retrieval" and expl.nearest_examples:
        tops = ", ".join([f"{e.get('merchant')} ({round(float(e.get('similarity',0))*100,1)}%)" for e in expl.nearest_examples])
        return base + f" Similar to: {tops}."
    return base


__all__ = ["build_explanation", "summarize"]
