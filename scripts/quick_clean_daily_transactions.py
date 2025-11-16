#!/usr/bin/env python3
"""Quick script to clean the Daily Household Transactions dataset."""

import pandas as pd
import json
from pathlib import Path

# Paths
ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = ROOT / "data/raw/daily-transactions-dataset/Daily Household Transactions.csv"
OUTPUT_CSV = ROOT / "data/merged_training.csv"
TAX_PATH = ROOT / "data/taxonomy.json"

# Load taxonomy
taxonomy = json.loads(TAX_PATH.read_text())
CANONICAL = [c["name"] for c in taxonomy["categories"]]

# Category normalization mapping
CATEGORY_MAP = {
    # From Daily Household Transactions subcategories
    "train": "Transportation",
    "snacks": "Food & Dining",
    "netflix": "Bills & Utilities",
    "mobile service provider": "Bills & Utilities",
    "ganesh pujan": "Entertainment",
    "tata play": "Bills & Utilities",
    "food": "Food & Dining",
    "restaurant": "Food & Dining",
    "groceries": "Food & Dining",
    "transport": "Transportation",
    "transportation": "Transportation",
    "taxi": "Transportation",
    "cab": "Transportation",
    "gas": "Transportation",
    "fuel": "Transportation",
    "petrol": "Transportation",
    "parking": "Transportation",
    "shopping": "Shopping",
    "apparel": "Shopping",
    "household": "Shopping",
    "gift": "Shopping",
    "beauty": "Shopping",
    "health": "Healthcare",
    "medical": "Healthcare",
    "hospital": "Healthcare",
    "doctor": "Healthcare",
    "pharmacy": "Healthcare",
    "travel": "Travel",
    "tourism": "Travel",
    "hotel": "Travel",
    "flight": "Travel",
    "vacation": "Travel",
    "entertainment": "Entertainment",
    "movie": "Entertainment",
    "movies": "Entertainment",
    "culture": "Entertainment",
    "festival": "Entertainment",
    "festivals": "Entertainment",
    "subscription": "Bills & Utilities",
    "utilities": "Bills & Utilities",
    "electricity": "Bills & Utilities",
    "water": "Bills & Utilities",
    "internet": "Bills & Utilities",
    "rent": "Bills & Utilities",
    "salary": "Income",
    "income": "Income",
    "interest": "Income",
    "dividend": "Income",
    "bonus": "Income",
    "investment": "Income",
    "maturity": "Income",
    "deposit": "Income",
    "refund": "Income",
}

def normalize_category(subcategory: str) -> str:
    """Normalize subcategory to canonical category."""
    if not subcategory:
        return "Uncategorized"
    
    s = str(subcategory).strip().lower()
    
    # Check exact match
    for canon in CANONICAL:
        if s == canon.lower():
            return canon
    
    # Check mapping
    if s in CATEGORY_MAP:
        return CATEGORY_MAP[s]
    
    # Check if any keyword in mapping appears in s
    for key, value in CATEGORY_MAP.items():
        if key in s or s in key:
            return value
    
    return "Uncategorized"

# Load and process
print("Loading Daily Household Transactions...")
df = pd.read_csv(INPUT_CSV)
print(f"  Loaded {len(df)} rows")

# Map columns: Note -> merchant, Amount -> amount, Subcategory -> category
cleaned = pd.DataFrame({
    "merchant": df["Note"].astype(str).fillna("").str.strip(),
    "amount": pd.to_numeric(df["Amount"], errors="coerce").fillna(0.0).abs(),
    "category": df["Subcategory"].apply(normalize_category)
})

# Filter out Uncategorized and empty merchants
before = len(cleaned)
cleaned = cleaned[(cleaned["category"] != "Uncategorized") & (cleaned["merchant"].str.len() > 0)]
after = len(cleaned)

print(f"  Cleaned: {after}/{before} rows kept ({100*after/before:.1f}%)")
print(f"\nCategory distribution:")
print(cleaned["category"].value_counts())

# Save
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
cleaned.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Saved to {OUTPUT_CSV.relative_to(ROOT)}")
print(f"  Total rows: {len(cleaned)}")
