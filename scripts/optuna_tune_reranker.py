"""Optuna hyperparameter tuning for Reranker XGBoost model.

This script uses a small training dataset to tune XGBoost hyperparameters and
saves the best calibrated model.
"""
from __future__ import annotations

import os
import optuna
import numpy as np
from typing import Dict, Any

from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

from src.reranker.model import Reranker
from src.storage.database import SessionLocal
from sqlalchemy import text
import json

def create_training_rows_from_eval(limit_per_cat=200):
    # Reuse the evaluation_results.json to synthesize training data
    import json
    p = 'evaluation_results.json'
    if not os.path.exists(p):
        raise RuntimeError('evaluation_results.json is required for this tuning script')
    with open(p,'r') as f:
        eval_data = json.load(f)
    rows = []
    categories = list(eval_data['per_class'].keys())
    # For each category, create synthetic examples
    for cat in categories:
        metrics = eval_data['per_class'][cat]
        n = min(limit_per_cat, max(50, metrics.get('support',50)))
        for i in range(n):
            merchant = f'{cat} Merchant {i}'
            amount = float(np.random.exponential(50) + 5)
            candidates = []
            for j in range(25):
                if j < 12:
                    cand_cat = cat
                    sim = float(np.random.beta(3,2))
                else:
                    cand_cat = np.random.choice(categories)
                    sim = float(np.random.beta(1,3))
                candidates.append({
                    'merchant': f'{merchant} {j}',
                    'amount': float(amount + np.random.normal(0,5)),
                    'category': cand_cat,
                    'similarity': float(sim)
                })
            rows.append({'query_text': merchant, 'candidates': candidates, 'label': cat})
    return rows


def create_training_rows_from_db(limit: int = 4000):
    db = SessionLocal()
    try:
        q = text('SELECT ge.id, ge.merchant, ge.amount, ge.description, gt.category_name FROM global_examples ge JOIN global_taxonomy gt ON ge.category_id = gt.id ORDER BY RANDOM() LIMIT :lim')
        rows = db.execute(q, {'lim': limit})
        examples = []
        for id_, merchant, amount, desc, cat in rows:
            candidates = []
            # for now we synthesize retrieval candidates via FAISS or randomization later
            # placeholder: use 25 random examples (for tuning purposes)
            for j in range(25):
                candidates.append({'merchant': merchant + f' cand {j}', 'amount': amount or 0.0, 'category': cat, 'similarity': 0.5})
            examples.append({'query_text': merchant, 'candidates': candidates, 'label': cat})
        return examples
    finally:
        db.close()


def objective(trial):
    # Hyperparameters to tune
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
        'min_child_weight': trial.suggest_float('min_child_weight', 1e-3, 10.0),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
    }

    reranker = Reranker()
    # Disable expensive CE scoring during optuna tuning for speed and avoid downloading
    reranker.has_ce = False
    rows = create_training_rows_from_eval(limit_per_cat=60)
    # Shuffle and split
    if len(rows) < 4:
        # Not enough samples to run a trial; return minimal score
        return 0.0
    train_rows, val_rows = train_test_split(rows, test_size=0.2, random_state=42)
    reranker.fit_xgb(train_rows, params=params, save=False)
    # Evaluate
    y_true = [r['label'] for r in val_rows]
    y_pred = []
    for r in val_rows:
        cat, conf, _ = reranker.rerank(r['query_text'], r['candidates'])
        y_pred.append(cat)
    # Compute macro F1 (mapping categories)
    macro = f1_score(y_true, y_pred, average='macro')
    # We want to maximize macro F1
    return macro


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--use-db', action='store_true', help='Use DB examples instead of evaluation JSON')
    parser.add_argument('--limit', type=int, default=4000, help='Limit number of DB examples for training')
    parser.add_argument('--precomputed-ce', default=None, help='Path to CE-augmented jsonl rows to use')
    args = parser.parse_args()

    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=args.trials)
    print('Best trial:', study.best_trial.params, study.best_value)
    # Train best model on full data and save
    best_params: Dict[str, Any] = study.best_trial.params
    reranker = Reranker()
    reranker.has_ce = False
    if args.use_db:
        rows = create_training_rows_from_db(limit=args.limit)
    else:
        rows = create_training_rows_from_eval(limit_per_cat=200)
    if args.precomputed_ce:
        # load CE-augmented rows
        augmented = []
        with open(args.precomputed_ce,'r') as f:
            for line in f:
                augmented.append(json.loads(line))
        if augmented:
            rows = augmented

    if not rows:
        raise RuntimeError('No training rows found. Provide DB rows with --use-db or precomputed CE rows with --precomputed-ce')
    if args.precomputed_ce:
        # load CE-augmented rows
        augmented = []
        with open(args.precomputed_ce,'r') as f:
            for line in f:
                augmented.append(json.loads(line))
        if augmented:
            rows = augmented
    reranker.fit_xgb(rows, params=best_params, save=True)
    print('Saved best model to', reranker.model_path)


if __name__ == '__main__':
    main()
