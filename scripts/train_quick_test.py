"""Quick training test for reranker to validate training pipeline with small data."""
from __future__ import annotations

import os
import numpy as np
import json
import sys
sys.path.insert(0, '/home/lathiss/Projects/LumaFin')
from src.reranker.model import Reranker


def create_small_training_rows(n=200):
    categories = ["Food", "Transport", "Shopping", "Entertainment"]
    rows = []
    for i in range(n):
        cat = np.random.choice(categories)
        merchant = f"{cat} Merchant {i}"
        amount = float(np.random.exponential(50) + 5)
        # generate candidates with majority in true category
        candidates = []
        for j in range(10):
            if j < 6:
                cand_cat = cat
                sim = np.random.beta(3, 2)
            else:
                cand_cat = np.random.choice(categories)
                sim = np.random.beta(1, 3)
            candidates.append({
                'merchant': f"{merchant} {j}",
                'amount': float(amount + np.random.normal(0, 5)),
                'category': cand_cat,
                'similarity': float(sim)
            })
        rows.append({ 'query_text': merchant, 'candidates': candidates, 'label': cat })
    return rows


def train_quick():
    rows = create_small_training_rows(200)
    reranker = Reranker()
    params = { 'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1, 'n_jobs': -1 }
    reranker.fit_xgb(rows, params=params, save=True)
    print("Saved model to: ", reranker.model_path)
    return reranker


if __name__ == '__main__':
    train_quick()
