"""Quick fusion evaluation on a small sample of CSV data.

This will load a few rows and run `decide()` for each, computing macro F1
using the project's `precision_recall_f1` util for consistent metrics.
"""
from __future__ import annotations

import csv
import argparse
from typing import List, Dict, Any
from src.fusion.decision import decide
from src.utils.metrics import precision_recall_f1


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
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()

    items = read_csv(args.csv, limit=args.limit)
    y_true = []
    y_pred = []
    for it in items:
        cat, conf, expl = decide({'merchant': it['merchant'], 'amount': it['amount'], 'description': it.get('description'), 'user_id': None})
        y_true.append(it['label'])
        y_pred.append(cat)

    metrics = precision_recall_f1(y_true, y_pred)
    result_path = 'evaluation_results_small.json'
    try:
        import json
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2)
        print('Saved metrics to', result_path)
    except Exception:
        print('Macro F1:', metrics['macro_avg']['f1'])
        print('Per-class F1:')
        for k, m in sorted(metrics.items()):
            if k in ('macro_avg', 'micro_avg'):
                continue
            print(k, '-> F1:', m['f1'], 'Support:', m['support'])


if __name__ == '__main__':
    main()
