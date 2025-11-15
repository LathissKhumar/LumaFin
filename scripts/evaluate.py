"""Simple evaluation script using retrieval baseline.

Usage:
  PYTHONPATH=. python scripts/evaluate.py --source db
  PYTHONPATH=. python scripts/evaluate.py --csv data/eval.csv
"""
from __future__ import annotations

import argparse
from typing import List

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.storage.database import SessionLocal
from src.embedder.encoder import TransactionEmbedder
from src.retrieval.service import get_retrieval_service
from src.utils.metrics import precision_recall_f1


def load_from_db(limit: int = 500) -> List[dict]:
    db: Session = SessionLocal()
    try:
        rows = db.execute(text(
            """
            SELECT ge.text, gt.category_name
            FROM global_examples ge
            JOIN global_taxonomy gt ON ge.category_id = gt.category_id
            ORDER BY ge.example_id
            LIMIT :lim
            """
        ), {"lim": limit}).fetchall()
        return [{"merchant": r[0], "amount": 0.0, "description": None, "label": r[1]} for r in rows]
    finally:
        db.close()


def evaluate(items: List[dict]) -> None:
    embedder = TransactionEmbedder()
    retriever = get_retrieval_service()
    y_true = []
    y_pred = []
    for it in items:
        emb = embedder.encode_transaction(it["merchant"], float(it.get("amount", 0.0)), it.get("description"))
        res = retriever.retrieve_by_embedding(emb, k=10)
        # vote simple majority
        votes = {}
        for r in res:
            votes[r["category"]] = votes.get(r["category"], 0) + 1
        pred = max(votes.items(), key=lambda kv: kv[1])[0] if votes else "Uncategorized"
        y_true.append(it["label"]) 
        y_pred.append(pred)
    metrics = precision_recall_f1(y_true, y_pred)
    print("Macro F1:", round(metrics["macro_avg"]["f1"], 4))
    print("Per-class:")
    for c, m in metrics.items():
        if c in ("macro_avg", "micro_avg"):
            continue
        print(c, "->", {k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()})


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["db"], default="db")
    ap.add_argument("--limit", type=int, default=500)
    args = ap.parse_args()
    if args.source == "db":
        items = load_from_db(args.limit)
    else:
        items = []
    evaluate(items)
