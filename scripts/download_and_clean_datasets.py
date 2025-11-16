#!/usr/bin/env python3
"""
Download and clean public transaction datasets into a consistent schema.

Outputs cleaned CSVs under data/processed/ and a merged file data/merged_training.csv
with columns: merchant,amount,category

Usage examples:
  # Download and clean specific Kaggle datasets by ID
  PYTHONPATH=. python scripts/download_and_clean_datasets.py \
    --dataset prasad22/daily-transactions-dataset \
    --dataset faizaniftikharjanjua/metaverse-financial-transactions-dataset

  # Clean already-downloaded CSV files or folders (no downloads)
  PYTHONPATH=. python scripts/download_and_clean_datasets.py --input data/raw/my_dataset

Notes:
- Requires Kaggle CLI if using --dataset. Ensure kaggle.json credentials are configured.
- Any dataset without the strict required columns (merchant, amount, category) 
  after auto-mapping is skipped.
- Rows with category 'Uncategorized' are dropped.
"""

import argparse
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import json

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROC_DIR = DATA_DIR / "processed"
MERGED_PATH = DATA_DIR / "merged_training.csv"

REQUIRED_COLS = ["merchant", "amount", "category"]

# Candidate column names for auto-mapping
MERCHANT_CANDIDATES = [
    "merchant", "merchant_name", "vendor", "payee", "name", "narration",
    "description", "text", "details", "memo", "counterparty", "note",
]
AMOUNT_CANDIDATES = [
    "amount", "amt", "value", "price", "transaction_amount", "txn_amount",
    "debit", "credit", "withdrawal", "deposit"
]
CATEGORY_CANDIDATES = [
    "category", "label", "class", "type", "expense_type", "merchant_category",
    "subcategory",
]

# Canonical categories
CANONICAL_CATEGORIES = [
    "Food & Dining",
    "Transportation",
    "Shopping",
    "Entertainment",
    "Bills & Utilities",
    "Healthcare",
    "Travel",
    "Income",
    "Uncategorized",
]


def load_taxonomy() -> Dict:
    tax_path = DATA_DIR / "taxonomy.json"
    if not tax_path.exists():
        raise FileNotFoundError("data/taxonomy.json not found")
    return json.loads(tax_path.read_text())


def normalize_category(raw: str, taxonomy: Dict) -> str:
    if not raw:
        return "Uncategorized"
    s = str(raw).strip().lower()

    # exact mapping to canonical
    for cat in CANONICAL_CATEGORIES:
        if s == cat.lower():
            return cat

    # explicit maps from common public datasets
    explicit = {
        # Food
        "food": "Food & Dining",
        "groceries": "Food & Dining",
        "grocery": "Food & Dining",
        "restaurants": "Food & Dining",
        "restaurant": "Food & Dining",
        # Shopping
        "shopping": "Shopping",
        "apparel": "Shopping",
        "household": "Shopping",
        "beauty": "Shopping",
        "gift": "Shopping",
        "electronics": "Shopping",
        # Entertainment
        "entertainment": "Entertainment",
        "movies": "Entertainment",
        "movie": "Entertainment",
        "culture": "Entertainment",
        "festival": "Entertainment",
        # Bills
        "utilities": "Bills & Utilities",
        "utility": "Bills & Utilities",
        "subscription": "Bills & Utilities",
        "internet": "Bills & Utilities",
        "rent": "Bills & Utilities",
        # Healthcare
        "health": "Healthcare",
        "medical": "Healthcare",
        # Travel
        "travel": "Travel",
        "tourism": "Travel",
        "transportation": "Transportation",
        # Income
        "income": "Income",
        "salary": "Income",
        "interest": "Income",
        "dividend": "Income",
        # Uncategorized-like
        "other": "Uncategorized",
        "uncategorized": "Uncategorized",
    }
    if s in explicit:
        return explicit[s]

    # keyword fallback using taxonomy
    keywords_map = {}
    for c in taxonomy.get("categories", []):
        kws = [k.lower() for k in c.get("keywords", [])]
        if any(k in s for k in kws):
            return c["name"]

    return "Uncategorized"


def map_columns(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    cols = {c.lower(): c for c in df.columns}

    def find_col(candidates: List[str]) -> Optional[str]:
        for cand in candidates:
            if cand in cols:
                return cols[cand]
        return None

    mcol = find_col(MERCHANT_CANDIDATES)
    acol = find_col(AMOUNT_CANDIDATES)
    ccol = find_col(CATEGORY_CANDIDATES)

    # handle debit/credit if amount missing
    if not acol:
        debit = cols.get("debit")
        credit = cols.get("credit")
        if debit or credit:
            df["__amount__"] = (
                (pd.to_numeric(df.get(debit, 0), errors="coerce").fillna(0).astype(float))
                - (pd.to_numeric(df.get(credit, 0), errors="coerce").fillna(0).astype(float))
            )
            acol = "__amount__"

    if not (mcol and acol and ccol):
        return None

    out = pd.DataFrame({
        "merchant": df[mcol].astype(str).fillna("").str.strip(),
        "amount": pd.to_numeric(df[acol], errors="coerce").fillna(0.0).astype(float),
        "category": df[ccol].astype(str).fillna("").str.strip(),
    })
    return out


def download_kaggle_dataset(dataset_id: str, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        print(f"↓ Downloading {dataset_id} ...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset_id, "-p", str(out_dir), "--unzip"],
            check=True,
        )
    except Exception as e:
        print(f"⚠️  Kaggle download failed for {dataset_id}: {e}")


def iter_csv_paths(root: Path) -> List[Path]:
    paths: List[Path] = []
    if root.is_file() and root.suffix.lower() == ".csv":
        return [root]
    for p in root.rglob("*.csv"):
        paths.append(p)
    return paths


def clean_inputs(inputs: List[Path], merge_out: Path) -> Tuple[int, int, int]:
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    taxonomy = load_taxonomy()

    total_rows = 0
    kept_rows = 0
    files_kept = 0

    merged_rows: List[pd.DataFrame] = []

    for inp in inputs:
        try:
            df = pd.read_csv(inp)
        except Exception as e:
            print(f"⚠️  Skipping {inp.name}: failed to read CSV ({e})")
            continue

        mapped = map_columns(df)
        if mapped is None:
            print(f"⚠️  Skipping {inp.name}: required columns not found")
            continue

        total_rows += len(mapped)

        # normalize categories
        mapped["category"] = mapped["category"].apply(lambda x: normalize_category(x, taxonomy))

        # drop Uncategorized and empty merchant
        cleaned = mapped[(mapped["category"] != "Uncategorized") & (mapped["merchant"].str.len() > 0)]
        keep_count = len(cleaned)
        if keep_count == 0:
            print(f"⚠️  Skipping {inp.name}: no rows after cleaning")
            continue

        # write individual cleaned file
        stem = inp.stem.replace(" ", "_")
        out_path = PROC_DIR / f"clean_{stem}.csv"
        cleaned.to_csv(out_path, index=False)
        print(f"✓ Cleaned {inp.name}: kept {keep_count}/{len(mapped)} rows → {out_path.relative_to(ROOT)}")

        kept_rows += keep_count
        files_kept += 1
        merged_rows.append(cleaned[REQUIRED_COLS])

    if merged_rows:
        merged_df = pd.concat(merged_rows, ignore_index=True)
        merged_df.to_csv(merge_out, index=False)
        print(f"\n✓ Merged {len(merged_rows)} cleaned files → {merge_out.relative_to(ROOT)}")
        print(f"  Total cleaned rows: {len(merged_df)}")
    else:
        print("\n⚠️  No cleaned files produced. Check input datasets and required columns.")

    return total_rows, kept_rows, files_kept


def main():
    parser = argparse.ArgumentParser(description="Download and clean transaction datasets")
    parser.add_argument("--dataset", action="append", default=[], help="Kaggle dataset ID to download (repeatable)")
    parser.add_argument("--input", action="append", default=[], help="Existing CSV file or folder to clean (repeatable)")
    parser.add_argument("--skip-download", action="store_true", help="Skip Kaggle downloads; only clean inputs")
    parser.add_argument("--output", default=str(MERGED_PATH), help="Merged output CSV path")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    inputs: List[Path] = []

    # 1) Download Kaggle datasets
    if not args.skip_download:
        for ds in args.dataset:
            slug = ds.split("/")[-1]
            out_dir = RAW_DIR / slug
            download_kaggle_dataset(ds, out_dir)
            inputs.extend(iter_csv_paths(out_dir))

    # 2) Add provided inputs
    for inp in args.input:
        p = Path(inp)
        if not p.exists():
            print(f"⚠️  Input not found: {inp}")
            continue
        inputs.extend(iter_csv_paths(p))

    # 3) Clean and merge
    total, kept, files = clean_inputs(inputs, Path(args.output))

    print("\n==== Summary ====")
    print(f"Inputs scanned: {len(inputs)} files")
    print(f"Rows scanned:   {total}")
    print(f"Rows kept:      {kept}")
    print(f"Cleaned files:  {files}")
    print(f"Merged output:  {args.output}")


if __name__ == "__main__":
    sys.exit(main())
