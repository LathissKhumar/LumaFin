"""Train the XGBoost reranker on labeled transactions with retrieval candidates.

Usage:
  PYTHONPATH=. python scripts/train_reranker.py --source db --limit 1000
  PYTHONPATH=. python scripts/train_reranker.py --csv data/training.csv
"""
from __future__ import annotations

import argparse
from typing import List, Dict, Any

from sqlalchemy import text
from sqlalchemy.orm import Session
import csv
import os

from src.storage.database import SessionLocal
from src.embedder.encoder import TransactionEmbedder
from src.retrieval.service import get_retrieval_service
from src.reranker.model import get_reranker
from src.utils.logger import get_logger


log = get_logger("train_reranker")


def load_from_db(limit: int = 1000) -> List[Dict[str, Any]]:
    """Load labeled examples from database."""
    db: Session = SessionLocal()
    try:
        rows = db.execute(text(
            """
            SELECT ge.merchant, ge.amount, ge.description, gt.category_name, ge.id
            FROM global_examples ge
            JOIN global_taxonomy gt ON ge.category_id = gt.id
            ORDER BY ge.id
            LIMIT :lim
            """
        ), {"lim": limit}).fetchall()
        return [
            {
                "merchant": r[0],
                "amount": float(r[1]) if r[1] else 0.0,
                "description": r[2],
                "label": r[3],
                "id": r[4]
            } for r in rows
        ]
    finally:
        db.close()


def prepare_training_rows(items: List[Dict[str, Any]], k: int = 20) -> List[Dict[str, Any]]:
    """Convert labeled items into training rows with retrieval candidates.
    
    Each training row contains:
      - query_text: merchant name
      - candidates: list of retrieval results (merchant, amount, category, similarity)
      - label: true category
    """
    embedder = TransactionEmbedder()
    retriever = get_retrieval_service()
    
    training_rows = []
    log.info(f"Preparing training rows for {len(items)} items...")
    
    for idx, it in enumerate(items):
        if idx % 100 == 0:
            log.info(f"Processed {idx}/{len(items)}")
        
        emb = embedder.encode_transaction(
            it["merchant"],
            float(it.get("amount", 0.0)),
            it.get("description")
        )
        
        # Get retrieval candidates (excluding self if possible)
        candidates = retriever.retrieve_by_embedding(emb, k=k+1)
        
        # Filter out the example itself if it appears in results
        example_id = it.get("id")
        if example_id:
            candidates = [c for c in candidates if c.get("id") != example_id]
        
        # Take top k
        candidates = candidates[:k]
        
        if candidates:
            training_rows.append({
                "query_text": it["merchant"],
                "candidates": candidates,
                "label": it["label"]
            })
    
    log.info(f"Prepared {len(training_rows)} training rows")
    return training_rows

def load_from_csv(paths: List[str]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for p in paths:
        if not p or not os.path.exists(p):
            log.warning(f"Skipping missing CSV path: {p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = {c.lower(): c for c in reader.fieldnames or []}
            merchant_col = cols.get("merchant") or cols.get("text") or cols.get("description")
            category_col = cols.get("category") or cols.get("label")
            amount_col = cols.get("amount")
            if not merchant_col or not category_col:
                log.warning(f"CSV {p} missing required columns; need merchant/text and category/label")
                continue
            for row in reader:
                merchant = row.get(merchant_col, "").strip()
                if not merchant:
                    continue
                label = row.get(category_col, "Uncategorized") or "Uncategorized"
                amt = 0.0
                if amount_col and row.get(amount_col):
                    try:
                        amt = float(row.get(amount_col))
                    except Exception:
                        amt = 0.0
                out.append({"merchant": merchant, "amount": amt, "description": None, "label": label, "id": None})
        log.info(f"Loaded {len(out)} cumulative rows after {p}")
    return out


def train_reranker(training_rows: List[Dict[str, Any]], params: Dict[str, Any] | None = None) -> None:
    """Train and save the XGBoost reranker."""
    reranker = get_reranker()
    
    if not reranker.has_xgb:
        log.error("XGBoost not available. Install with: pip install xgboost")
        return
    
    log.info(f"Training reranker on {len(training_rows)} examples...")
    
    # Default params optimized for small datasets
    default_params = {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.07,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "n_jobs": -1,  # Use all cores
    }
    
    cfg = {**default_params, **(params or {})}
    log.info(f"Training params: {cfg}")
    
    # Train and save
    reranker.fit_xgb(training_rows, params=cfg, save=True)
    
    log.info(f"✅ Training complete! Model saved to {reranker.model_path}")
    log.info("The reranker will automatically use this model in future predictions.")


def main():
    ap = argparse.ArgumentParser(description="Train XGBoost reranker for transaction categorization")
    ap.add_argument("--source", choices=["db", "csv"], default="db", help="Data source")
    ap.add_argument("--limit", type=int, default=1000, help="Number of examples to use")
    ap.add_argument("--k", type=int, default=20, help="Number of retrieval candidates per query")
    ap.add_argument("--n-estimators", type=int, default=200, help="Number of XGBoost trees")
    ap.add_argument("--max-depth", type=int, default=4, help="Max tree depth")
    ap.add_argument("--learning-rate", type=float, default=0.07, help="Learning rate")
    ap.add_argument("--csv", nargs="*", help="CSV files to use when source=csv")
    args = ap.parse_args()
    
    # Load data
    log.info(f"Loading data from {args.source}...")
    if args.source == "db":
        items = load_from_db(args.limit)
    elif args.source == "csv":
        if not args.csv:
            log.error("--csv files required when source=csv")
            return
        items = load_from_csv(list(args.csv))
    else:
        items = []
    
    if not items:
        log.error("No training data loaded!")
        return
    
    log.info(f"Loaded {len(items)} labeled examples")
    
    # Prepare training rows
    training_rows = prepare_training_rows(items, k=args.k)
    
    if not training_rows:
        log.error("No training rows prepared!")
        return
    
    # Train
    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
    }
    train_reranker(training_rows, params=params)


if __name__ == "__main__":
    main()
