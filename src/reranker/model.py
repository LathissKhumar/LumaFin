"""Reranker module.

Provides a reranker abstraction. If cross-encoder / xgboost are available, they
can be plugged in. By default, falls back to a heuristic using retrieval
similarities and simple votes. Includes lightweight feature engineering so a
drop-in XGBoost model can be trained later without changing the API.
"""
from __future__ import annotations

from typing import List, Dict, Any, Tuple
import math
import os

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
        self.model_path = os.getenv("RERANKER_MODEL_PATH", "models/reranker/xgb_reranker.json")
        if self.has_ce:
            try:
                self.ce_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
                self.ce_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception:
                # Fallback to heuristic if model can't be loaded (e.g., offline)
                self.has_ce = False
        if self.has_xgb:
            try:
                import xgboost as xgb  # type: ignore
                if os.path.exists(self.model_path):
                    self.xgb_model = xgb.XGBClassifier()
                    self.xgb_model.load_model(self.model_path)
            except Exception:
                self.xgb_model = None
        self._is_trained = self.xgb_model is not None

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

    def _bucket(self, v: float, edges: List[float]) -> int:
        for i, e in enumerate(edges):
            if v < e:
                return i
        return len(edges)

    def _features_per_category(self, query_text: str, candidates: List[Dict[str, Any]]) -> Dict[str, List[float]]:
        """Engineer features per category from retrieval candidates.

        Features per category:
          0: count (top-K frequency)
          1: sum_similarity
          2: max_similarity
          3: mean_similarity
          4: min_similarity
          5: vote_fraction (count/K)
          6: amount_diff_min (log1p proximity, if amounts present)
        """
        K = max(1, len(candidates))
        by_cat: Dict[str, Dict[str, Any]] = {}
        # Prepare query amount if any appears in text (naive parse)
        def _amount_from_text(txt: str) -> float | None:
            return None  # placeholder – often amount is separate; we rely on candidate amounts
        q_amount = _amount_from_text(query_text)
        for c in candidates:
            cat = c.get("category") or c.get("category_name") or "Unknown"
            sim = float(c.get("similarity", 0.0))
            amt = c.get("amount")
            entry = by_cat.setdefault(cat, {"count": 0, "sum": 0.0, "max": -1.0, "min": 2.0, "amount_diffs": []})
            entry["count"] += 1
            entry["sum"] += sim
            entry["max"] = max(entry["max"], sim)
            entry["min"] = min(entry["min"], sim)
            if amt is not None:
                try:
                    a = float(amt)
                    if q_amount is not None:
                        entry["amount_diffs"].append(abs(math.log1p(q_amount) - math.log1p(a)))
                    else:
                        entry["amount_diffs"].append(0.0)
                except Exception:
                    pass
        feats: Dict[str, List[float]] = {}
        for cat, d in by_cat.items():
            count = float(d["count"])
            s = float(d["sum"]) if count else 0.0
            mx = float(d["max"]) if count else 0.0
            mn = float(d["min"]) if count else 0.0
            mean = s / count if count else 0.0
            vf = count / float(K)
            amt_min = min(d["amount_diffs"]) if d["amount_diffs"] else 0.0
            feats[cat] = [count, s, mx, mean, mn, vf, amt_min]
        return feats

    def _xgb_predict(self, feats: Dict[str, List[float]]) -> Tuple[str, float]:
        import numpy as np  # local import
        cats = list(feats.keys())
        X = np.array([feats[c] for c in cats], dtype=float)
        # If model not present, fallback
        if not self.xgb_model:
            # score by weighted combination similar to heuristic
            scores = {c: v[2] + 0.2 * v[3] + 0.1 * v[5] for c, v in feats.items()}  # max_sim + 0.2*mean + 0.1*vote_frac
            best = max(scores.items(), key=lambda kv: kv[1])
            conf = 0.5 + 0.4 * max(0.0, min(1.0, float(best[1])))
            return best[0], conf
        # Predict probabilities per row; choose highest
        try:
            proba = self.xgb_model.predict_proba(X)
            idx = int(proba[:, 1].argmax()) if proba.shape[1] > 1 else int(proba[:, 0].argmax())
            p = float(proba[idx, 1] if proba.shape[1] > 1 else proba[idx, 0])
            return cats[idx], p
        except Exception:
            # Fallback to heuristic if prediction fails
            scores = {c: v[2] + 0.2 * v[3] + 0.1 * v[5] for c, v in feats.items()}
            best = max(scores.items(), key=lambda kv: kv[1])
            conf = 0.5 + 0.4 * max(0.0, min(1.0, float(best[1])))
            return best[0], conf

    def fit_xgb(self, rows: List[Dict[str, Any]], params: Dict[str, Any] | None = None, save: bool = True) -> None:
        """Train an XGBoost reranker from labeled rows.

        Each row must be a dict with:
          - query_text: str
          - candidates: List[Dict[str, Any]]  # as returned by retrieval
          - label: str  # true category for the query

        We construct per-category feature vectors for each query, assigning y=1 to
        the true label row and y=0 to the rest, then train a binary classifier.
        """
        if not _HAS_XGB:
            return
        import numpy as np  # type: ignore
        import xgboost as xgb  # type: ignore

        X_list: List[List[float]] = []
        y_list: List[int] = []
        for row in rows:
            q = row.get("query_text", "")
            cands = row.get("candidates", [])
            y_true = row.get("label")
            feats = self._features_per_category(q, cands)
            for cat, vec in feats.items():
                X_list.append(vec)
                y_list.append(1 if cat == y_true else 0)
        if not X_list:
            return
        X = np.array(X_list, dtype=float)
        y = np.array(y_list, dtype=int)
        # Default, simple params suitable for small data
        default_params = {
            "n_estimators": 200,
            "max_depth": 4,
            "learning_rate": 0.07,
            "subsample": 0.9,
            "colsample_bytree": 0.8,
            "reg_lambda": 1.0,
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "n_jobs": 1,
        }
        cfg = {**default_params, **(params or {})}
        model = xgb.XGBClassifier(**cfg)
        model.fit(X, y)
        self.xgb_model = model
        self._is_trained = True
        if save:
            # Ensure directory exists
            try:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                model.save_model(self.model_path)
            except Exception:
                pass

    def rerank(self, query_text: str, candidates: List[Dict[str, Any]]) -> Tuple[str, float, List[Dict[str, Any]]]:
        """Return (category, confidence, enriched_candidates).

        candidates item schema expected keys: merchant, amount, category, similarity
        """
        if not candidates:
            return "Uncategorized", 0.0, []
        # If we have feature-based scorer (XGB), use it; else heuristic
        feats = self._features_per_category(query_text, candidates)
        if feats:
            cat, conf = self._xgb_predict(feats)
            return cat, conf, candidates
        # Heuristic fallback
        cat, conf = self._heuristic_score(candidates)
        return cat, conf, candidates


_global_reranker: Reranker | None = None


def get_reranker() -> Reranker:
    global _global_reranker
    if _global_reranker is None:
        _global_reranker = Reranker()
    return _global_reranker


__all__ = ["Reranker", "get_reranker"]
