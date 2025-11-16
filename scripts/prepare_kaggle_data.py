"""
Prepare Kaggle datasets for training by producing a unified CSV with
columns: merchant, amount, category.

Usage:
  PYTHONPATH=. python scripts/prepare_kaggle_data.py --input data/kaggle --output data/merged_training.csv

The script tries to map common column names and will use the rules engine
for weak labeling when category/label column is missing.
"""
from __future__ import annotations

import argparse
import csv
import os
from typing import List, Dict, Optional

from src.preprocessing.normalize import normalize_merchant
from src.rules.engine import RuleEngine


MERCHANT_CANDIDATE_COLS = [
    "merchant", "text", "description", "name", "narration", "memo", "payee", "details", "merchant_name",
    "note", "subcategory", "transaction_type"
]
AMOUNT_CANDIDATE_COLS = [
    "amount", "amt", "value", "price", "transaction_amount", "debit", "credit"
]
CATEGORY_CANDIDATE_COLS = [
    "category", "label", "type", "group"
]


def choose_col(cols: List[str], fieldnames: List[str]) -> Optional[str]:
    lower = {c.lower(): c for c in fieldnames}
    for c in cols:
        if c.lower() in lower:
            return lower[c.lower()]
    return None


def process_csv(path: str, engine: RuleEngine) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            return rows
        merch_col = choose_col(MERCHANT_CANDIDATE_COLS, reader.fieldnames)
        amt_col = choose_col(AMOUNT_CANDIDATE_COLS, reader.fieldnames)
        cat_col = choose_col(CATEGORY_CANDIDATE_COLS, reader.fieldnames)

        if not merch_col:
            # Can't use this file
            return rows

        for r in reader:
            raw_m = (r.get(merch_col) or "").strip()
            if not raw_m:
                continue
            m = normalize_merchant(raw_m)
            # Parse amount
            amt_val = 0.0
            if amt_col and r.get(amt_col):
                try:
                    amt_val = float(str(r.get(amt_col)).replace(",", ""))
                except Exception:
                    amt_val = 0.0

            # Determine category
            cat = None
            if cat_col and r.get(cat_col):
                cat = str(r.get(cat_col)).strip()
            if not cat:
                # Weakly label via rules
                c = engine.apply_rules(m, amt_val)
                cat = c.name if c else "Uncategorized"

            rows.append({
                "merchant": m,
                "amount": f"{amt_val}",
                "category": cat
            })
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/kaggle", help="Directory containing CSV files")
    ap.add_argument("--output", default="data/merged_training.csv", help="Output CSV path")
    args = ap.parse_args()

    engine = RuleEngine()

    merged: List[Dict[str, str]] = []
    if os.path.isdir(args.input):
        for root, _dirs, files in os.walk(args.input):
            for fn in files:
                if fn.lower().endswith(".csv"):
                    path = os.path.join(root, fn)
                    rows = process_csv(path, engine)
                    print(f"Processed {len(rows)} rows from {path}")
                    merged.extend(rows)
    else:
        if args.input.lower().endswith(".csv"):
            merged.extend(process_csv(args.input, engine))

    # Write out
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["merchant", "amount", "category"])
        writer.writeheader()
        writer.writerows(merged)

    print(f"Wrote {len(merged)} rows to {args.output}")


if __name__ == "__main__":
    main()
