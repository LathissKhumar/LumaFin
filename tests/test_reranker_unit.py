from __future__ import annotations

import numpy as np
from src.reranker.model import Reranker


def test_heuristic_score_empty_candidates():
    reranker = Reranker()
    cat, conf, enriched = reranker.rerank("test", [])
    assert cat == "Uncategorized"
    assert conf == 0.0
    assert enriched == []


def test_heuristic_score_single_category():
    reranker = Reranker()
    candidates = [
        {"category": "Food", "similarity": 0.9, "merchant": "Restaurant A", "amount": 20.0},
        {"category": "Food", "similarity": 0.85, "merchant": "Restaurant B", "amount": 25.0},
    ]
    cat, conf, enriched = reranker.rerank("restaurant", candidates)
    assert cat == "Food"
    assert 0.4 <= conf <= 1.0
    assert len(enriched) == 2


def test_feature_engineering():
    reranker = Reranker()
    candidates = [
        {"category": "Food", "similarity": 0.9, "merchant": "Rest A", "amount": 20.0},
        {"category": "Food", "similarity": 0.8, "merchant": "Rest B", "amount": 25.0},
        {"category": "Transport", "similarity": 0.7, "merchant": "Uber", "amount": 15.0},
    ]
    feats = reranker._features_per_category("restaurant", candidates)
    assert "Food" in feats
    assert "Transport" in feats
    food_feats = feats["Food"]
    assert food_feats[0] == 2.0
    assert abs(food_feats[1] - 1.7) < 0.01
    assert abs(food_feats[2] - 0.9) < 0.01
    assert abs(food_feats[3] - 0.85) < 0.01
    assert abs(food_feats[5] - 2/3) < 0.01
    assert len(food_feats) == 14
    assert abs(food_feats[11] - 1.7) < 0.01
    transport_feats = feats["Transport"]
    assert transport_feats[0] == 1.0
    assert len(transport_feats) == 14


def test_xgb_fallback_scoring():
    reranker = Reranker()
    feats = {
        "CatA": [2.0, 1.5, 0.9, 0.75, 0.6, 0.6, 0.0, 0.9, 0.0, 10.0, 0.05, 1.7, 0.8, 0.7],
        "CatB": [3.0, 1.8, 0.4, 0.6, 0.2, 0.75, 0.0, 0.1, 0.0, 20.0, 0.25, 1.1, 0.05, 0.04],
    }
    cat, conf = reranker._xgb_predict(feats)
    assert cat == "CatA"
    assert 0.0 <= conf <= 1.0
