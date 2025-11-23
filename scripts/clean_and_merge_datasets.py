"""
Clean and merge all downloaded datasets into a unified training CSV.

Output format:
- merchant: str (normalized)
- amount: float
- category: str (mapped to taxonomy)
- description: str (optional)
- label: str (optional short user label)
- date: str (YYYY-MM-DD format)

Target: 80,000+ clean rows
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
import re
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from src.preprocessing.normalize import normalize_merchant

RAW_DATA_DIR = Path("data/raw")
OUTPUT_FILE = Path("data/merged_training.csv")
TAXONOMY_FILE = Path("data/taxonomy.json")


def load_taxonomy() -> dict:
    """Load canonical taxonomy."""
    if not TAXONOMY_FILE.exists():
        print(f"✗ Taxonomy file not found: {TAXONOMY_FILE}")
        print("Creating default taxonomy...")
        taxonomy = {
            "categories": [
                {"name": "Food & Dining", "keywords": ["food", "restaurant", "cafe", "grocery", "dining"]},
                {"name": "Shopping", "keywords": ["shop", "retail", "store", "amazon", "walmart"]},
                {"name": "Transportation", "keywords": ["transport", "uber", "lyft", "gas", "parking"]},
                {"name": "Entertainment", "keywords": ["entertainment", "movie", "netflix", "spotify", "game"]},
                {"name": "Bills & Utilities", "keywords": ["bill", "utility", "electric", "internet", "rent"]},
                {"name": "Healthcare", "keywords": ["health", "medical", "doctor", "pharmacy", "hospital"]},
                {"name": "Travel", "keywords": ["travel", "hotel", "flight", "airbnb", "vacation"]},
                {"name": "Income", "keywords": ["income", "salary", "deposit", "refund", "dividend"]},
                {"name": "Uncategorized", "keywords": []}
            ]
        }
        TAXONOMY_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TAXONOMY_FILE, 'w') as f:
            json.dump(taxonomy, f, indent=2)
    
    with open(TAXONOMY_FILE, 'r') as f:
        return json.load(f)


def map_to_canonical_category(raw_category: str, merchant: str, taxonomy: dict) -> str:
    """Map raw category to canonical taxonomy category."""
    if not raw_category or pd.isna(raw_category):
        raw_category = "Uncategorized"
    
    raw_lower = str(raw_category).lower().strip()
    merchant_lower = str(merchant).lower().strip()
    
    # Build keyword map
    keyword_map = {}
    for cat in taxonomy["categories"]:
        for kw in cat["keywords"]:
            keyword_map[kw] = cat["name"]
    
    # Canonical category names
    canonical_names = [c["name"] for c in taxonomy["categories"]]
    
    # 1. Exact match
    for name in canonical_names:
        if raw_lower == name.lower():
            return name
    
    # 2. Explicit mappings
    explicit_maps = {
        "food": "Food & Dining",
        "groceries": "Food & Dining",
        "restaurants": "Food & Dining",
        "gas": "Transportation",
        "fuel": "Transportation",
        "automotive": "Transportation",
        "streaming": "Entertainment",
        "subscriptions": "Bills & Utilities",
        "personal": "Shopping",
        "cash": "Uncategorized",
        "transfer": "Uncategorized",
        "misc": "Uncategorized",
        "other": "Uncategorized",
    }
    
    for key, value in explicit_maps.items():
        if key in raw_lower:
            return value
    
    # 3. Keyword matching on merchant name
    for kw, cat_name in keyword_map.items():
        if kw in merchant_lower:
            return cat_name
    
    # 4. Keyword matching on category
    for kw, cat_name in keyword_map.items():
        if kw in raw_lower:
            return cat_name
    
    return "Uncategorized"


def clean_amount(amount_str) -> float:
    """Clean and convert amount to float."""
    if pd.isna(amount_str):
        return 0.0
    
    if isinstance(amount_str, (int, float)):
        return abs(float(amount_str))
    
    # Remove currency symbols and commas
    amount_str = str(amount_str).replace('$', '').replace(',', '').replace('£', '').replace('€', '').strip()
    
    try:
        return abs(float(amount_str))
    except:
        return 0.0


def clean_date(date_str) -> str:
    """Convert date to YYYY-MM-DD format."""
    if pd.isna(date_str):
        return datetime.now().strftime("%Y-%m-%d")
    
    try:
        # Try parsing various formats
        for fmt in ["%Y-%m-%d", "%m/%d/%Y", "%d/%m/%Y", "%Y/%m/%d", "%m-%d-%Y"]:
            try:
                dt = datetime.strptime(str(date_str), fmt)
                return dt.strftime("%Y-%m-%d")
            except:
                continue
        
        # Try pandas
        dt = pd.to_datetime(date_str)
        return dt.strftime("%Y-%m-%d")
    except:
        return datetime.now().strftime("%Y-%m-%d")


def detect_columns(df: pd.DataFrame) -> dict:
    """Detect which columns map to our schema."""
    cols = {c.lower(): c for c in df.columns}
    
    mapping = {}
    
    # Merchant/description column
    for key in ["merchant", "description", "trans_description", "memo", "name", "payee"]:
        if key in cols:
            mapping["merchant"] = cols[key]
            break
    
    # Amount column
    for key in ["amount", "amt", "value", "price", "total", "transaction_amount", "debit"]:
        if key in cols:
            mapping["amount"] = cols[key]
            break
    
    # Category column
    for key in ["category", "cat", "type", "transaction_type", "expense_type"]:
        if key in cols:
            mapping["category"] = cols[key]
            break
    
    # Date column
    for key in ["date", "trans_date", "transaction_date", "timestamp", "time"]:
        if key in cols:
            mapping["date"] = cols[key]
            break
    
    return mapping


def clean_dataset(df: pd.DataFrame, source_name: str, taxonomy: dict) -> pd.DataFrame:
    """Clean and standardize a single dataset."""
    print(f"  Cleaning {source_name}: {len(df)} rows")
    
    # Detect column mapping
    col_map = detect_columns(df)
    
    if "merchant" not in col_map or "amount" not in col_map:
        print(f"  ✗ Skipping {source_name}: missing required columns")
        return pd.DataFrame()
    
    # Extract and clean data
    cleaned_data = []
    
    for idx, row in df.iterrows():
        merchant_raw = row.get(col_map["merchant"], "")
        amount_raw = row.get(col_map.get("amount", "amount"), 0)
        category_raw = row.get(col_map.get("category", "category"), "Uncategorized") if "category" in col_map else "Uncategorized"
        date_raw = row.get(col_map.get("date", "date"), "") if "date" in col_map else ""
        
        # Skip if merchant is empty
        if pd.isna(merchant_raw) or str(merchant_raw).strip() == "":
            continue
        
        # Clean merchant
        merchant = normalize_merchant(str(merchant_raw))
        if not merchant or len(merchant) < 2:
            continue
        
        # Clean amount
        amount = clean_amount(amount_raw)
        if amount <= 0 or amount > 100000:  # Filter unrealistic amounts
            continue
        
        # Map category
        category = map_to_canonical_category(category_raw, merchant, taxonomy)
        
        # Clean date
        date = clean_date(date_raw)
        
        cleaned_data.append({
            "merchant": merchant,
            "amount": amount,
            "category": category,
            "date": date,
            "description": merchant,  # Use merchant as description
            "label": ""  # Empty label for now
        })
    
    cleaned_df = pd.DataFrame(cleaned_data)
    print(f"  ✓ Cleaned to {len(cleaned_df)} rows")
    return cleaned_df


def main():
    print("="*60)
    print("LUMAFIN DATASET CLEANING & MERGING")
    print("="*60)
    print()
    
    # Load taxonomy
    print("Loading taxonomy...")
    taxonomy = load_taxonomy()
    print(f"✓ Loaded {len(taxonomy['categories'])} categories")
    print()
    
    # Find all raw CSV files
    print("Scanning for raw datasets...")
    raw_files = list(RAW_DATA_DIR.glob("**/*.csv"))
    print(f"Found {len(raw_files)} CSV files")
    print()
    
    if not raw_files:
        print("✗ No CSV files found in data/raw/")
        print("Run scripts/download_large_datasets.py first")
        return
    
    # Clean each dataset
    print("Cleaning datasets...")
    all_cleaned = []
    
    for csv_file in raw_files:
        try:
            print(f"\nProcessing: {csv_file}")
            df = pd.read_csv(csv_file, low_memory=False)
            print(f"  Loaded {len(df)} rows, {len(df.columns)} columns")
            print(f"  Columns: {list(df.columns)[:5]}...")
            cleaned = clean_dataset(df, csv_file.name, taxonomy)
            if len(cleaned) > 0:
                all_cleaned.append(cleaned)
        except Exception as e:
            print(f"  ✗ Error processing {csv_file.name}: {e}")
            import traceback
            traceback.print_exc()
    
    print()
    
    # Merge all datasets
    print("Merging datasets...")
    if not all_cleaned:
        print("✗ No datasets were successfully cleaned")
        return
    
    merged_df = pd.concat(all_cleaned, ignore_index=True)
    print(f"✓ Merged to {len(merged_df)} total rows")
    print()
    
    # If we don't have enough, generate more synthetic data
    if len(merged_df) < 80000:
        shortage = 80000 - len(merged_df)
        print(f"Generating {shortage} additional synthetic transactions to reach target...")
        from scripts.download_large_datasets import generate_synthetic_transactions
        synthetic_df = generate_synthetic_transactions(shortage)
        # Clean synthetic data
        synthetic_clean = clean_dataset(synthetic_df, "synthetic_additional", taxonomy)
        if len(synthetic_clean) > 0:
            all_cleaned.append(synthetic_clean)
            merged_df = pd.concat([merged_df, synthetic_clean], ignore_index=True)
            print(f"✓ Added {len(synthetic_clean)} synthetic rows")
            print(f"New total: {len(merged_df)} rows")
        print()
    
    # Remove duplicates
    print("Removing duplicates...")
    initial_count = len(merged_df)
    merged_df = merged_df.drop_duplicates(subset=["merchant", "amount", "date"], keep="first")
    print(f"✓ Removed {initial_count - len(merged_df)} duplicates")
    print(f"Final count: {len(merged_df)} rows")
    print()
    
    # Category distribution
    print("Category distribution:")
    cat_counts = merged_df["category"].value_counts()
    for cat, count in cat_counts.items():
        pct = (count / len(merged_df)) * 100
        print(f"  {cat:20s}: {count:6d} ({pct:5.1f}%)")
    print()
    
    # Save merged dataset
    print(f"Saving to {OUTPUT_FILE}...")
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Saved {len(merged_df)} rows")
    print()
    
    # Check if target met
    if len(merged_df) >= 80000:
        print(f"✅ Target achieved: {len(merged_df):,} rows (>= 80,000)")
    else:
        print(f"⚠️  Target not met: {len(merged_df):,} rows (need 80,000)")
        print("   Consider running download script again or generating more synthetic data")
    
    print()
    print("="*60)
    print("Next step: Seed database with merged_training.csv")
    print("  PYTHONPATH=. python scripts/seed_database.py --csv data/merged_training.csv")
    print("="*60)


if __name__ == "__main__":
    main()
