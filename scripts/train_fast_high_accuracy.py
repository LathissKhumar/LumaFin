"""Fast Training Pipeline for >90% Accuracy

Uses optimized hyperparameters and efficient training to achieve high accuracy quickly.
"""
from __future__ import annotations

import os
import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import classification_report, f1_score
import xgboost as xgb

from src.storage.database import SessionLocal
from src.embedder.encoder import TransactionEmbedder
from src.retrieval.service import get_retrieval_service
from src.reranker.model import Reranker
from src.utils.logger import get_logger

# Disable tokenizers parallelism
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

log = get_logger("fast_training")


def load_training_data(limit: int = 3000) -> List[Dict[str, Any]]:
    """Load training data from database."""
    from sqlalchemy import text
    from sqlalchemy.orm import Session

    db = SessionLocal()
    try:
        rows = db.execute(text("""
            SELECT ge.id, ge.merchant, ge.amount, ge.description, gt.category_name
            FROM global_examples ge
            JOIN global_taxonomy gt ON ge.category_id = gt.id
            ORDER BY RANDOM()
            LIMIT :lim
        """), {"lim": limit}).fetchall()

        examples = []
        for row in rows:
            examples.append({
                'id': row[0],
                'merchant': row[1],
                'amount': float(row[2]) if row[2] else 0.0,
                'description': row[3],
                'category': row[4],
            })

        log.info(f"Loaded {len(examples)} training examples")
        return examples

    finally:
        db.close()


def prepare_training_data(examples: List[Dict[str, Any]], k: int = 25) -> List[Dict[str, Any]]:
    """Prepare training data with retrieval candidates."""
    embedder = TransactionEmbedder()
    retriever = get_retrieval_service()

    training_rows = []

    for i, example in enumerate(examples):
        if i % 200 == 0:
            log.info(f"Processing {i}/{len(examples)} examples")

        # Get embedding
        emb = embedder.encode_transaction(
            example['merchant'], example['amount'], example['description']
        )

        # Get candidates
        candidates = retriever.retrieve_by_embedding(emb, k=k+1)
        candidates = [c for c in candidates if c.get('id') != example.get('id')][:k]

        if candidates:
            training_rows.append({
                'query_text': example['merchant'],
                'candidates': candidates,
                'label': example['category']
            })

    log.info(f"Prepared {len(training_rows)} training rows")
    return training_rows


def train_high_accuracy_reranker():
    """Train reranker with optimized parameters for >90% accuracy."""

    # Load and prepare data
    examples = load_training_data(3000)
    training_rows = prepare_training_data(examples, k=25)

    # Optimized hyperparameters for high accuracy
    params = {
        'n_estimators': 800,
        'max_depth': 8,
        'learning_rate': 0.05,
        'subsample': 0.9,
        'colsample_bytree': 0.9,
        'reg_alpha': 0.1,
        'reg_lambda': 0.5,
        'objective': 'multi:softprob',
        'eval_metric': 'mlogloss',
        'n_jobs': -1,
        'random_state': 42,
    }

    # Initialize reranker and train
    reranker = Reranker()

    log.info("Training high-accuracy reranker...")
    log.info(f"Using parameters: {params}")

    reranker.fit_xgb(training_rows, params=params, save=True)

    # Evaluate on training data
    log.info("Evaluating training performance...")

    correct = 0
    total = 0

    for row in training_rows[:500]:  # Evaluate on subset for speed
        category, confidence, _ = reranker.rerank(row['query_text'], row['candidates'])
        if category == row['label']:
            correct += 1
        total += 1

    accuracy = correct / total * 100
    log.info(".2f")

    if accuracy >= 90.0:
        print("🎉 SUCCESS: Achieved >90% training accuracy!")
    elif accuracy >= 85.0:
        print("👍 GOOD: Achieved >85% training accuracy")
    else:
        print("⚠️  NEEDS IMPROVEMENT: Training accuracy below 85%")

    return accuracy


def run_full_evaluation():
    """Run comprehensive evaluation on the trained model."""
    from scripts.evaluate import evaluate

    log.info("Running full evaluation...")
    results = evaluate(mode='fusion', limit=1000)

    macro_f1 = results['macro']['f1'] * 100
    micro_f1 = results['micro']['f1'] * 100

    print("\n" + "="*60)
    print("FINAL EVALUATION RESULTS")
    print("="*60)
    print(".2f")
    print(".2f")

    if macro_f1 >= 90.0:
        print("🎉 SUCCESS: Achieved >90% macro F1 accuracy!")
        return True
    elif macro_f1 >= 85.0:
        print("👍 GOOD: Achieved >85% macro F1 accuracy")
        return True
    else:
        print("⚠️  NEEDS IMPROVEMENT: Macro F1 below 85%")
        return False

    print("\nTop 5 performing categories:")
    sorted_cats = sorted(results['per_class'].items(),
                        key=lambda x: x[1]['f1'], reverse=True)
    for cat, metrics in sorted_cats[:5]:
        f1 = metrics['f1'] * 100
        support = metrics['support']
        print(".1f")

    return macro_f1 >= 85.0


def main():
    print("🚀 Fast Training Pipeline for >90% Accuracy")
    print("="*50)

    try:
        # Train the model
        train_accuracy = train_high_accuracy_reranker()

        # Run evaluation
        success = run_full_evaluation()

        if success:
            print("\n✅ Training pipeline completed successfully!")
            print("Model saved and ready for production use.")
        else:
            print("\n❌ Training pipeline completed but accuracy target not met.")
            print("Consider increasing training data or adjusting hyperparameters.")

    except Exception as e:
        log.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())