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
import numpy as np

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
                # Allow disabling CE load via env var to speed up tests / tuning
                if os.getenv('RERANKER_DISABLE_CE', '0') == '1':
                    self.has_ce = False
                else:
                    self.ce_tokenizer = AutoTokenizer.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
                    self.ce_model = AutoModelForSequenceClassification.from_pretrained("cross-encoder/ms-marco-MiniLM-L-6-v2")
            except Exception:
                # Fallback to heuristic if model can't be loaded (e.g., offline)
                self.has_ce = False
        if self.has_xgb:
            try:
                import xgboost as xgb  # type: ignore
                if os.path.exists(self.model_path):
                    # Support both xgboost native JSON model and pickled sklearn models
                    if self.model_path.endswith('.pkl'):
                        try:
                            import joblib
                            self.xgb_model = joblib.load(self.model_path)
                        except Exception:
                            self.xgb_model = None
                    else:
                        try:
                            self.xgb_model = xgb.XGBClassifier()
                            self.xgb_model.load_model(self.model_path)
                        except Exception:
                            # Fallback to None if JSON load fails
                            self.xgb_model = None
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

    def _features_per_category(self, query_text: str, candidates: List[Dict[str, Any]], query_label: str | None = None, hour_of_day: int | None = None, weekday: int | None = None) -> Dict[str, List[float]]:
        """Engineer enhanced features per category from retrieval candidates.

        Enhanced features per category (12 total):
          0: count (top-K frequency)
          1: sum_similarity
          2: max_similarity
          3: mean_similarity
          4: min_similarity
          5: vote_fraction (count/K)
          6: amount_diff_min (log1p proximity)
          7: merchant_similarity (Jaccard similarity of merchant names)
          8: category_distribution_entropy (diversity of categories in candidates)
          9: amount_range (max - min amount in category)
          10: similarity_std (standard deviation of similarities)
          11: top_3_similarity_sum (sum of top 3 similarities)
        """
        K = max(1, len(candidates))
        by_cat: Dict[str, Dict[str, Any]] = {}

        # Extract query merchant and amount
        query_merchant = query_text.lower().strip()
        query_amount = None
        for c in candidates:
            if 'amount' in c and c['amount'] is not None:
                try:
                    query_amount = float(c['amount'])
                    break
                except (ValueError, TypeError):
                    continue

        for c in candidates:
            cat = c.get("category") or c.get("category_name") or "Unknown"
            sim = float(c.get("similarity", 0.0))
            amt = c.get("amount")
            merchant = c.get("merchant", "").lower().strip()

            entry = by_cat.setdefault(cat, {
                "count": 0, "sum": 0.0, "max": -1.0, "min": 2.0,
                "amounts": [], "similarities": [], "merchants": []
            })
            entry["count"] += 1
            entry["sum"] += sim
            entry["max"] = max(entry["max"], sim)
            entry["min"] = min(entry["min"], sim)
            entry["similarities"].append(sim)
            entry["merchants"].append(merchant)

            if amt is not None:
                try:
                    entry["amounts"].append(float(amt))
                except (ValueError, TypeError):
                    pass

        # Convert to feature vectors
        features = {}
        all_cats = list(by_cat.keys())

        for cat, stats in by_cat.items():
            count = stats["count"]
            if count == 0:
                continue

            similarities = stats["similarities"]
            mean_sim = stats["sum"] / count
            vote_frac = count / K

            # Amount features
            amounts = stats["amounts"]
            amount_diff_min = 0.0
            amount_range = 0.0
            if amounts and query_amount is not None:
                diffs = [abs(a - query_amount) for a in amounts]
                amount_diff_min = min(diffs) if diffs else 0.0
                amount_range = max(amounts) - min(amounts) if len(amounts) > 1 else 0.0
            elif amounts:
                amount_range = max(amounts) - min(amounts) if len(amounts) > 1 else 0.0

            # Merchant similarity (Jaccard)
            merchant_sim = 0.0
            if query_merchant and stats["merchants"]:
                query_words = set(query_merchant.split())
                if query_words:
                    cat_merchants = [set(m.split()) for m in stats["merchants"] if m]
                    if cat_merchants:
                        similarities = []
                        for cat_words in cat_merchants:
                            intersection = len(query_words & cat_words)
                            union = len(query_words | cat_words)
                            if union > 0:
                                similarities.append(intersection / union)
                        merchant_sim = max(similarities) if similarities else 0.0

            # Category distribution entropy
            cat_dist_entropy = 0.0
            if len(all_cats) > 1:
                cat_counts = [by_cat[c]["count"] for c in all_cats]
                total = sum(cat_counts)
                if total > 0:
                    probs = [cnt/total for cnt in cat_counts]
                    cat_dist_entropy = -sum(p * math.log(p) for p in probs if p > 0)

            # Similarity statistics
            similarity_std = np.std(similarities) if len(similarities) > 1 else 0.0
            top_3_sim_sum = sum(sorted(similarities, reverse=True)[:3])

            features[cat] = [
                float(count),           # 0: count
                stats["sum"],           # 1: sum_similarity
                stats["max"],           # 2: max_similarity
                mean_sim,               # 3: mean_similarity
                stats["min"],           # 4: min_similarity
                vote_frac,              # 5: vote_fraction
                math.log1p(amount_diff_min) if amount_diff_min > 0 else 0.0,  # 6: amount_diff_min (log scaled)
                merchant_sim,           # 7: merchant_similarity
                cat_dist_entropy,       # 8: category_distribution_entropy
                amount_range,           # 9: amount_range
                similarity_std,         # 10: similarity_std
                top_3_sim_sum,          # 11: top_3_similarity_sum
            ]
            # Cross-encoder features: include max and mean CE score per category
            ce_scores = [c.get("ce_score", 0.0) for c in candidates if (c.get("category") or c.get("category_name") or "Unknown") == cat]
            if ce_scores:
                features[cat].append(max(ce_scores))   # 12: max_ce_score
                features[cat].append(sum(ce_scores) / len(ce_scores))  # 13: mean_ce_score
            else:
                features[cat].append(0.0)
                features[cat].append(0.0)
            # Label-based features (query_label might be None)
            # label_exact_match (1.0 if candidate label equals query label else 0.0)
            label_votes = [c.get("label") for c in candidates if c.get("category") == cat and c.get("label")]
            label_exact_match = 0.0
            label_vote_fraction = 0.0
            if query_label:
                normalized_q_label = query_label.lower().strip()
                label_matches = [1 for l in label_votes if l and l.lower().strip() == normalized_q_label]
                if label_votes:
                    label_vote_fraction = sum(label_matches) / len(label_votes)
                    if sum(label_matches) > 0:
                        label_exact_match = 1.0
            features[cat].append(float(label_exact_match))  # 14: label_exact_match
            features[cat].append(float(label_vote_fraction))  # 15: label_vote_fraction

        return features

    def _xgb_predict(self, feats: Dict[str, List[float]]) -> Tuple[str, float]:
        import numpy as np  # local import
        cats = list(feats.keys())
        X = np.array([feats[c] for c in cats], dtype=float)
        # If model not present, fallback
        if not self.xgb_model:
            # Enhanced heuristic scoring with new features
            scores = {}
            for c, v in feats.items():
                # Weighted combination: max_sim + 0.2*mean + 0.1*vote_frac + 0.3*merchant_sim + 0.1*top_3_sum
                score = (v[2] + 0.2 * v[3] + 0.1 * v[5] + 0.3 * v[7] + 0.1 * v[11] + 0.2 * (v[12] if len(v) > 12 else 0.0))
                scores[c] = score
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
            # Enhanced fallback heuristic if prediction fails
            scores = {}
            for c, v in feats.items():
                score = (v[2] + 0.2 * v[3] + 0.1 * v[5] + 0.3 * v[7] + 0.1 * v[11] + 0.2 * (v[12] if len(v) > 12 else 0.0))
                scores[c] = score
            best = max(scores.items(), key=lambda kv: kv[1])
            conf = 0.5 + 0.4 * max(0.0, min(1.0, float(best[1])))
            return best[0], conf

    def _score_with_ce(self, query_text: str, candidates: List[Dict[str, Any]]) -> List[float]:
        """Compute cross-encoder scores for a list of candidates with the query.

        Returns a list of floats in the same order as candidates. If cross-encoder
        is unavailable, returns zeros.
        """
        if not self.has_ce or self.ce_model is None or self.ce_tokenizer is None:
            return [0.0 for _ in candidates]
        try:
            import torch
            texts1 = [query_text] * len(candidates)
            texts2 = [c.get('merchant', '') for c in candidates]
            # Tokenize pairs
            inputs = self.ce_tokenizer(texts1, texts2, padding=True, truncation=True, return_tensors='pt')
            inputs = {k: v.to(self.ce_model.device) for k, v in inputs.items()}
            with torch.no_grad():
                out = self.ce_model(**inputs)
                logits = out.logits
                # If regression-like model with single logit per pair
                if logits.dim() == 1 or logits.shape[1] == 1:
                    logits = logits.reshape(-1)
                    scores = torch.sigmoid(logits).cpu().numpy().tolist()
                else:
                    # Multiclass; take first column and sigmoid
                    scores = torch.sigmoid(logits[:, 0]).cpu().numpy().tolist()
            return [float(s) for s in scores]
        except Exception:
            return [0.0 for _ in candidates]

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
        # If training labels are single-class (all 0 or all 1), skip training
        unique_vals = set(y.tolist())
        if len(unique_vals) < 2:
            import logging
            logging.getLogger(__name__).warning(
                "Reranker XGBoost training skipped: single-class labels found in training data: %s",
                sorted(unique_vals)
            )
            # Ensure model remains None and allow fallback heuristic
            self.xgb_model = None
            self._is_trained = False
            return
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
        # Split for calibration
        from sklearn.model_selection import train_test_split
        try:
            X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        except Exception:
            # Fallback to no stratify if y has single class
            X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.2, random_state=42)
        model.fit(X_train, y_train)
        # Calibrate with Platt sigmoid using held-out calibration set
        try:
            from sklearn.calibration import CalibratedClassifierCV
            clf = CalibratedClassifierCV(base_estimator=model, method='sigmoid', cv='prefit')
            clf.fit(X_cal, y_cal)
            self.xgb_model = clf
        except Exception:
            # If calibration fails, fall back to raw model
            self.xgb_model = model
        self._is_trained = True
        if save:
            # Ensure directory exists
            try:
                os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
                # Save as pickle (supports calibrated sklearn objects)
                try:
                    import joblib
                    joblib.dump(self.xgb_model, self.model_path)
                except Exception:
                    # Fall back to xgboost save if model is xgboost native
                    try:
                        model.save_model(self.model_path)
                    except Exception:
                        pass
            except Exception:
                pass

    def rerank(self, query_text: str, candidates: List[Dict[str, Any]], query_label: str | None = None, hour_of_day: int | None = None, weekday: int | None = None) -> Tuple[str, float, List[Dict[str, Any]]]:
        """Return (category, confidence, enriched_candidates).

        candidates item schema expected keys: merchant, amount, category, similarity
        """
        if not candidates:
            return "Uncategorized", 0.0, []
        # Attach cross-encoder scores (if available) to candidates
        try:
            ce_scores = self._score_with_ce(query_text, candidates)
            for c, s in zip(candidates, ce_scores):
                c['ce_score'] = float(s)
        except Exception:
            for c in candidates:
                c['ce_score'] = 0.0

        # If we have feature-based scorer (XGB), use it; else heuristic
        feats = self._features_per_category(query_text, candidates, query_label=query_label, hour_of_day=hour_of_day, weekday=weekday)
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
