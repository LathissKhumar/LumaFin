"""Train the XGBoost reranker using CE-augmented training rows (JSONL).

This script loads CE-augmented rows from JSONL, trains the XGB reranker with
calibration, and saves the model to `models/reranker/ce_trained.pkl` by default.
It accepts a number of command-line args to control params and output.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List, Dict, Any

from src.reranker.model import Reranker


def load_rows_from_jsonl(path: str, limit: int | None = None) -> List[Dict[str, Any]]:
    rows = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default='data/rows_ce.jsonl', help='CE-augmented JSONL rows')
    parser.add_argument('--model-path', default='models/reranker/ce_trained.pkl')
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--n-estimators', type=int, default=800)
    parser.add_argument('--max-depth', type=int, default=8)
    parser.add_argument('--learning-rate', type=float, default=0.05)
    args = parser.parse_args()

    print('[train_reranker_with_ce] Loading rows from', args.input)
    rows = load_rows_from_jsonl(args.input, limit=args.limit)
    if not rows:
        print('[train_reranker_with_ce] No rows found - aborting')
        return 1

    params = {
        'n_estimators': args.n_estimators,
        'max_depth': args.max_depth,
        'learning_rate': args.learning_rate,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.1,
        'reg_lambda': 0.5,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'n_jobs': -1,
        'random_state': 42,
    }

    print(f"[train_reranker_with_ce] Training with {len(rows)} rows, params={params}")
    reranker = Reranker()
    # Ensure we use CE during training
    reranker.has_ce = True
    reranker.fit_xgb(rows, params=params, save=True)
    # Save to specified path if different
    if args.model_path and args.model_path != reranker.model_path:
        try:
            import joblib
            os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
            joblib.dump(reranker.xgb_model, args.model_path)
            print('[train_reranker_with_ce] Model saved to', args.model_path)
        except Exception:
            print('[train_reranker_with_ce] Warning: failed to save to', args.model_path)
    else:
        print('[train_reranker_with_ce] Model saved to default path', reranker.model_path)

    return 0


if __name__ == '__main__':
    exit(main())
