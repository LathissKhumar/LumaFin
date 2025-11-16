"""Unit tests for LumaFin core components.

Tests cover:
- Fusion decision pipeline
- Reranker feature engineering
- Personal centroid matching
- Rule engine
- Embedding consistency

Run:
  PYTHONPATH=. pytest tests/test_core.py -v
"""
from __future__ import annotations

import pytest
import numpy as np
from typing import Dict, Any, List

from src.fusion.decision import decide, _scale_centroid_conf
from src.reranker.model import Reranker
from src.clustering.centroid_matcher import SIMILARITY_THRESHOLD
from src.rules.engine import RuleResult
from src.embedder.encoder import TransactionEmbedder


class TestFusionPipeline:
    """Test hierarchical decision pipeline."""
    
    def test_scale_centroid_confidence(self):
        """Test confidence scaling for centroid matches."""
        # At threshold, should be base (0.85)
        conf = _scale_centroid_conf(SIMILARITY_THRESHOLD, base=0.85, max_c=0.95)
        assert 0.84 <= conf <= 0.86
        
        # At 1.0 similarity, should be max_c (0.95)
        conf = _scale_centroid_conf(1.0, base=0.85, max_c=0.95)
        assert 0.94 <= conf <= 0.96
        
        # Below threshold, should still return base
        conf = _scale_centroid_conf(0.5, base=0.85, max_c=0.95)
        assert conf >= 0.85
    
    def test_fusion_fallback(self):
        """Test fallback when no rules or centroids match."""
        txn = {
            "merchant": "Unknown Merchant XYZ",
            "amount": 99.99,
            "description": None,
            "user_id": None
        }
        cat, conf, expl = decide(txn)
        # Without rules/centroids/retrieval hits, should fallback
        assert expl["decision_path"] in ("retrieval", "fallback")
        assert 0.0 <= conf <= 1.0


class TestReranker:
    """Test reranker feature engineering and scoring."""
    
    def test_heuristic_score_empty_candidates(self):
        """Test reranker with no candidates."""
        reranker = Reranker()
        cat, conf, enriched = reranker.rerank("test", [])
        assert cat == "Uncategorized"
        assert conf == 0.0
        assert enriched == []
    
    def test_heuristic_score_single_category(self):
        """Test reranker with single category candidates."""
        reranker = Reranker()
        candidates = [
            {"category": "Food", "similarity": 0.9, "merchant": "Restaurant A", "amount": 20.0},
            {"category": "Food", "similarity": 0.85, "merchant": "Restaurant B", "amount": 25.0},
        ]
        cat, conf, enriched = reranker.rerank("restaurant", candidates)
        assert cat == "Food"
        assert 0.4 <= conf <= 1.0
        assert len(enriched) == 2
    
    def test_feature_engineering(self):
        """Test per-category feature extraction."""
        reranker = Reranker()
        candidates = [
            {"category": "Food", "similarity": 0.9, "merchant": "Rest A", "amount": 20.0},
            {"category": "Food", "similarity": 0.8, "merchant": "Rest B", "amount": 25.0},
            {"category": "Transport", "similarity": 0.7, "merchant": "Uber", "amount": 15.0},
        ]
        feats = reranker._features_per_category("restaurant", candidates)
        
        assert "Food" in feats
        assert "Transport" in feats
        
        # Food should have: count=2, sum=1.7, max=0.9, mean=0.85, min=0.8, vote_frac=2/3
        food_feats = feats["Food"]
        assert food_feats[0] == 2.0  # count
        assert abs(food_feats[1] - 1.7) < 0.01  # sum
        assert abs(food_feats[2] - 0.9) < 0.01  # max
        assert abs(food_feats[3] - 0.85) < 0.01  # mean
        assert abs(food_feats[5] - 2/3) < 0.01  # vote fraction
        # Check that additional features exist and have expected types
        assert len(food_feats) == 14
        # Top-3 similarity sum should be 0.9 + 0.8 = 1.7 (since only 2 food candidates)
        assert abs(food_feats[11] - 1.7) < 0.01
        
        # Transport should have: count=1
        transport_feats = feats["Transport"]
        assert transport_feats[0] == 1.0
        assert len(transport_feats) == 14

    def test_xgb_fallback_scoring(self):
        """Test that _xgb_predict fallback uses enhanced scoring to pick category."""
        reranker = Reranker()
        # Create two category feature sets: A has higher max_similarity and merchant_sim
        feats = {
            "CatA": [2.0, 1.5, 0.9, 0.75, 0.6, 0.6, 0.0, 0.9, 0.0, 10.0, 0.05, 1.7, 0.8, 0.7],
            "CatB": [3.0, 1.8, 0.4, 0.6, 0.2, 0.75, 0.0, 0.1, 0.0, 20.0, 0.25, 1.1, 0.05, 0.04],
        }
        cat, conf = reranker._xgb_predict(feats)
        assert cat == "CatA"
        assert 0.0 <= conf <= 1.0


class TestEmbedder:
    """Test embedding consistency and normalization."""
    
    def test_encode_transaction_deterministic(self):
        """Test that encoding same transaction gives same embedding."""
        embedder = TransactionEmbedder()
        emb1 = embedder.encode_transaction("Starbucks", 5.50, "Coffee")
        emb2 = embedder.encode_transaction("Starbucks", 5.50, "Coffee")
        
        # Should be deterministic
        assert np.allclose(emb1, emb2, rtol=1e-5)
    
    def test_encode_transaction_normalized(self):
        """Test that embeddings are normalized to unit length."""
        embedder = TransactionEmbedder()
        emb = embedder.encode_transaction("Test Merchant", 100.0)
        
        # Should be normalized (L2 norm ≈ 1)
        norm = np.linalg.norm(emb)
        assert 0.99 <= norm <= 1.01
    
    def test_encode_batch(self):
        """Test batch encoding."""
        embedder = TransactionEmbedder()
        transactions = [
            {"merchant": "Store A", "amount": 10.0, "description": "Purchase"},
            {"merchant": "Store B", "amount": 20.0, "description": None},
        ]
        embeddings = embedder.encode_batch(transactions)
        
        assert embeddings.shape[0] == 2
        assert embeddings.shape[1] == 384  # MiniLM dimension


class TestRuleEngine:
    """Test deterministic rule matching."""
    
    def test_rule_result_creation(self):
        """Test RuleResult dataclass."""
        result = RuleResult(
            name="TestCategory",
            priority=1,
            pattern="test.*",
            confidence=1.0
        )
        assert result.name == "TestCategory"
        assert result.confidence == 1.0
    
    def test_empty_rule_list(self):
        """Test that empty rules return None."""
        # This would require mocking the rule engine, skipping for now
        pass


class TestCentroidMatcher:
    """Test personal centroid matching logic."""
    
    def test_similarity_threshold(self):
        """Test that threshold is correctly defined."""
        assert SIMILARITY_THRESHOLD == 0.80
    
    def test_match_requires_min_support(self):
        """Test minimum support requirement for centroid matching."""
        # This would require database mocking, documenting expected behavior
        # A centroid match should only fire if:
        # 1. Similarity > 0.80
        # 2. Centroid has >= 5 supporting transactions
        pass


# Integration test markers
@pytest.mark.integration
class TestIntegration:
    """Integration tests requiring database and models."""
    
    def test_end_to_end_categorization(self):
        """Test full pipeline from transaction to category."""
        # This would test the full flow:
        # 1. Create transaction
        # 2. Call decide()
        # 3. Verify category, confidence, explanation
        pytest.skip("Requires live database and FAISS index")
    
    def test_feedback_loop(self):
        """Test feedback submission and centroid update."""
        pytest.skip("Requires live database and Celery")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
