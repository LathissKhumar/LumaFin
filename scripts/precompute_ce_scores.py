"""Precompute cross-encoder scores for training rows and save augmented data.

This script reads training rows (optionally from DB or evaluation JSON) and uses
the cross-encoder model to compute a score for each candidate pair
(query, candidate.merchant). The result is saved as a JSONL file with 'ce_score'
added to each candidate.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict, Any

from src.reranker.model import Reranker


def load_rows_from_eval(path: str) -> List[Dict[str, Any]]:
    with open(path, 'r') as f:
        data = json.load(f)
    # If 'per_class' -> produce rows as earlier for training
    if 'per_class' in data:
        categories = list(data['per_class'].keys())
        rows = []
        for cat in categories:
            metrics = data['per_class'][cat]
            support = metrics.get('support', 50)
            n = max(50, support)
            for i in range(n):
                merchant = f'{cat} Merchant {i}'
                amount = 5.0
                candidates = []
                for j in range(25):
                    if j < 12:
                        cand_cat = cat
                        sim = 0.8
                    else:
                        cand_cat = categories[(i+j) % len(categories)]
                        sim = 0.2
                    candidates.append({
                        'merchant': f'{merchant} {j}',
                        'amount': amount,
                        'category': cand_cat,
                        'similarity': float(sim),
                    })
                rows.append({'query_text': merchant, 'candidates': candidates, 'label': cat})
        return rows
    else:
        return data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', default='evaluation_results.json')
    parser.add_argument('--out', default='data/rows_with_ce.jsonl')
    args = parser.parse_args()

    rows = load_rows_from_eval(args.eval)
    r = Reranker()

    print('Computing CE scores for', len(rows), 'rows')
    with open(args.out, 'w') as out_f:
        for i, row in enumerate(rows):
            try:
                ce_scores = r._score_with_ce(row['query_text'], row['candidates'])
                for c, s in zip(row['candidates'], ce_scores):
                    c['ce_score'] = float(s)
            except Exception:
                for c in row['candidates']:
                    c['ce_score'] = 0.0
            json.dump(row, out_f)
            out_f.write('\n')
            if (i+1) % 100 == 0:
                print(f'Processed {i+1}/{len(rows)}')
    print('Saved CE-augmented rows to', args.out)


if __name__ == '__main__':
    main()
