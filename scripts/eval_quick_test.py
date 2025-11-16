"""Quick evaluation for the reranker model using a synthetic test set.

Loads the reranker model and evaluates accuracy on a synthetic dataset.
"""
from __future__ import annotations

import numpy as np
import sys
sys.path.insert(0, '/home/lathiss/Projects/LumaFin')
from src.reranker.model import Reranker


def create_test_rows(n=200):
    categories = ["Food", "Transport", "Shopping", "Entertainment"]
    rows = []
    for i in range(n):
        cat = np.random.choice(categories)
        merchant = f"{cat} Merchant {i}"
        amount = float(np.random.exponential(50) + 5)
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


def main():
    reranker = Reranker()
    rows = create_test_rows(500)
    correct = 0
    total = 0
    for row in rows:
        cat, conf, _ = reranker.rerank(row['query_text'], row['candidates'])
        if cat == row['label']:
            correct += 1
        total += 1
    print(f"Accuracy: {correct}/{total} = {correct/total*100:.2f}%")

if __name__ == '__main__':
    main()
