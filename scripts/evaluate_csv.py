"""Evaluate the full fusion pipeline using a CSV as input (no DB dependency).

This script loads labeled rows from a CSV and runs `evaluate()` in `scripts/evaluate.py`.
CSV must have columns merchant/text, category/label, amount(optional), description(optional).
"""
from __future__ import annotations

import argparse
import csv
from typing import List, Dict, Any

from scripts.evaluate import evaluate


def read_csv(path: str, limit: int | None = None) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        cols = {c.lower(): c for c in (reader.fieldnames or [])}
        merchant_col = cols.get('merchant') or cols.get('text') or cols.get('description')
        category_col = cols.get('category') or cols.get('label')
        amount_col = cols.get('amount')
        for i, row in enumerate(reader):
            if limit and i >= limit:
                break
            merchant = row.get(merchant_col, '') if merchant_col else ''
            category = row.get(category_col, '') if category_col else 'Uncategorized'
            amount = float(row.get(amount_col, 0.0)) if amount_col and row.get(amount_col) else 0.0
            items.append({
                'merchant': merchant,
                'amount': amount,
                'description': row.get('description'),
                'label': category,
            })
    return items


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='data/merged_training.csv')
    parser.add_argument('--limit', type=int, default=1000)
    parser.add_argument('--mode', default='fusion', choices=['retrieval', 'reranker', 'fusion'])
    parser.add_argument('--output', default='evaluation_results_csv.json')
    args = parser.parse_args()

    items = read_csv(args.csv, limit=args.limit)
    print(f"Running evaluation in mode {args.mode} with {len(items)} items")
    evaluate(items, mode=args.mode, output_path=args.output)


if __name__ == '__main__':
    main()
