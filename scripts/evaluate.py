"""Evaluation script with optional reranker and fusion.

Usage:
  PYTHONPATH=. python scripts/evaluate.py --source db --mode retrieval
  PYTHONPATH=. python scripts/evaluate.py --source db --mode reranker
  PYTHONPATH=. python scripts/evaluate.py --source db --mode fusion
"""
from __future__ import annotations

import argparse
import json
import os
from typing import List

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.storage.database import SessionLocal
from src.embedder.encoder import TransactionEmbedder
from src.retrieval.service import get_retrieval_service
from src.reranker.model import get_reranker
from src.fusion.decision import decide
from src.utils.metrics import precision_recall_f1


def load_from_db(limit: int = 500) -> List[dict]:
    db: Session = SessionLocal()
    try:
        rows = db.execute(text(
            """
            SELECT ge.merchant, ge.amount, gt.category_name
            FROM global_examples ge
            JOIN global_taxonomy gt ON ge.category_id = gt.id
            ORDER BY ge.id
            LIMIT :lim
            """
        ), {"lim": limit}).fetchall()
        return [{"merchant": r[0], "amount": float(r[1] or 0.0), "description": None, "label": r[2]} for r in rows]
    finally:
        db.close()


def evaluate(items: List[dict], mode: str = "retrieval", output_path: str = "evaluation_results.json") -> None:
    """Evaluate using retrieval baseline, reranker, or full fusion pipeline.
    
    Args:
        items: List of dicts with keys: merchant, amount, description, label
        mode: "retrieval" (baseline), "reranker" (retrieval+rerank), or "fusion" (full pipeline)
    """
    embedder = TransactionEmbedder()
    retriever = get_retrieval_service()
    reranker = get_reranker() if mode in ("reranker", "fusion") else None
    
    y_true = []
    y_pred = []
    
    for it in items:
        if mode == "fusion":
            # Use full fusion pipeline (rules, centroids, retrieval, reranker)
            txn = {
                "merchant": it["merchant"],
                "amount": float(it.get("amount", 0.0)),
                "description": it.get("description"),
                "user_id": None  # No user context in global eval
            }
            pred, conf, expl = decide(txn)
        else:
            # Retrieval or reranker mode
            emb = embedder.encode_transaction(it["merchant"], float(it.get("amount", 0.0)), it.get("description"))
            res = retriever.retrieve_by_embedding(emb, k=20)
            
            if mode == "reranker" and reranker and res:
                pred, conf, enriched = reranker.rerank(it["merchant"], res)
            else:
                # Baseline retrieval with vote majority
                votes = {}
                for r in res:
                    votes[r["category"]] = votes.get(r["category"], 0) + 1
                pred = max(votes.items(), key=lambda kv: kv[1])[0] if votes else "Uncategorized"
        
        y_true.append(it["label"]) 
        y_pred.append(pred)
    
    metrics = precision_recall_f1(y_true, y_pred)
    print(f"\n{'='*60}")
    print(f"Evaluation Mode: {mode.upper()}")
    print(f"{'='*60}")
    print(f"Macro F1: {round(metrics['macro_avg']['f1'], 4)}")
    print(f"Macro Precision: {round(metrics['macro_avg']['precision'], 4)}")
    print(f"Macro Recall: {round(metrics['macro_avg']['recall'], 4)}")
    print(f"\nPer-class metrics:")
    for c, m in sorted(metrics.items()):
        if c in ("macro_avg", "micro_avg"):
            continue
        print(f"  {c:20s} -> P: {round(m['precision'], 3):<5} R: {round(m['recall'], 3):<5} F1: {round(m['f1'], 3):<5} Support: {m['support']}")
    print(f"{'='*60}\n")

    # Persist results
    try:
        payload = {
            "mode": mode,
            "macro": metrics.get("macro_avg"),
            "micro": metrics.get("micro_avg"),
            "per_class": {k: v for k, v in metrics.items() if k not in ("macro_avg", "micro_avg")},
            "support_total": sum(v.get("support", 0) for k, v in metrics.items() if k not in ("macro_avg", "micro_avg")),
        }
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"Saved evaluation metrics to {output_path}")
    except Exception as e:
        print(f"Failed to write evaluation results: {e}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["db"], default="db")
    ap.add_argument("--limit", type=int, default=500)
    ap.add_argument("--mode", choices=["retrieval", "reranker", "fusion"], default="retrieval",
                    help="Evaluation mode: retrieval (baseline), reranker (with XGBoost), or fusion (full pipeline)")
    ap.add_argument("--output", default="evaluation_results.json", help="Path to save evaluation JSON")
    args = ap.parse_args()
    if args.source == "db":
        items = load_from_db(args.limit)
    else:
        items = []
    evaluate(items, mode=args.mode, output_path=args.output)
