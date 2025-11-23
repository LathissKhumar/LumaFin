"""
Validate merged dataset and clean up unnecessary files.

This script:
1. Validates merged_training.csv format and quality
2. Removes old/duplicate raw datasets  
3. Creates backup
4. Generates summary statistics
"""
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
import json
from datetime import datetime

MERGED_FILE = Path("data/merged_training.csv")
TAXONOMY_FILE = Path("data/taxonomy.json")
RAW_DIR = Path("data/raw")
BACKUP_DIR = Path("data/backups")
REPORT_FILE = Path("data/dataset_report.json")


def validate_dataset(df: pd.DataFrame) -> dict:
    """Validate dataset quality and return report."""
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_rows": len(df),
        "validation": {},
        "statistics": {},
        "warnings": []
    }
    
    # Check required columns
    required_cols = ["merchant", "amount", "category", "date", "description", "label"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    report["validation"]["required_columns"] = len(missing_cols) == 0
    if missing_cols:
        report["warnings"].append(f"Missing columns: {missing_cols}")
    
    # Check for nulls in critical columns
    null_merchants = df["merchant"].isna().sum()
    null_amounts = df["amount"].isna().sum()
    null_categories = df["category"].isna().sum()
    
    report["validation"]["no_null_merchants"] = null_merchants == 0
    report["validation"]["no_null_amounts"] = null_amounts == 0
    report["validation"]["no_null_categories"] = null_categories == 0
    
    if null_merchants > 0:
        report["warnings"].append(f"{null_merchants} rows with null merchants")
    if null_amounts > 0:
        report["warnings"].append(f"{null_amounts} rows with null amounts")
    if null_categories > 0:
        report["warnings"].append(f"{null_categories} rows with null categories")
    
    # Check amount validity
    invalid_amounts = ((df["amount"] <= 0) | (df["amount"] > 100000)).sum()
    report["validation"]["valid_amounts"] = invalid_amounts == 0
    if invalid_amounts > 0:
        report["warnings"].append(f"{invalid_amounts} rows with invalid amounts")
    
    # Statistics
    report["statistics"]["amount"] = {
        "min": float(df["amount"].min()),
        "max": float(df["amount"].max()),
        "mean": float(df["amount"].mean()),
        "median": float(df["amount"].median()),
        "std": float(df["amount"].std())
    }
    
    # Category distribution
    cat_dist = df["category"].value_counts().to_dict()
    report["statistics"]["category_distribution"] = {k: int(v) for k, v in cat_dist.items()}
    
    # Unique merchants
    report["statistics"]["unique_merchants"] = int(df["merchant"].nunique())
    
    # Date range
    try:
        df["date_parsed"] = pd.to_datetime(df["date"])
        report["statistics"]["date_range"] = {
            "min": df["date_parsed"].min().strftime("%Y-%m-%d"),
            "max": df["date_parsed"].max().strftime("%Y-%m-%d")
        }
    except:
        report["warnings"].append("Could not parse dates")
    
    return report


def cleanup_raw_data():
    """Remove or archive unnecessary raw datasets."""
    print("Cleaning up raw data directory...")
    
    if not RAW_DIR.exists():
        print("  No raw directory found")
        return
    
    # Create backup of raw data before cleanup
    backup_path = BACKUP_DIR / f"raw_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    print(f"  Creating backup at {backup_path}...")
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        shutil.copytree(RAW_DIR, backup_path)
        print(f"  ✓ Backup created")
    except Exception as e:
        print(f"  ⚠️  Backup failed: {e}")
        return
    
    # Now clean up raw directory (keep .gitkeep)
    for item in RAW_DIR.iterdir():
        if item.name == ".gitkeep":
            continue
        try:
            if item.is_dir():
                shutil.rmtree(item)
                print(f"  ✓ Removed directory: {item.name}")
            else:
                item.unlink()
                print(f"  ✓ Removed file: {item.name}")
        except Exception as e:
            print(f"  ✗ Could not remove {item.name}: {e}")
    
    print(f"  ✓ Raw data cleaned up (backup saved)")


def main():
    print("="*60)
    print("DATASET VALIDATION & CLEANUP")
    print("="*60)
    print()
    
    # 1. Load and validate dataset
    print(f"Loading {MERGED_FILE}...")
    df = pd.read_csv(MERGED_FILE)
    print(f"✓ Loaded {len(df):,} rows")
    print()
    
    print("Validating dataset...")
    report = validate_dataset(df)
    print(f"✓ Validation complete")
    print()
    
    # 2. Print validation results
    print("Validation Results:")
    all_valid = all(report["validation"].values())
    if all_valid:
        print("  ✅ All validations passed")
    else:
        for check, passed in report["validation"].items():
            status = "✓" if passed else "✗"
            print(f"  {status} {check}")
    print()
    
    if report["warnings"]:
        print("Warnings:")
        for warning in report["warnings"]:
            print(f"  ⚠️  {warning}")
        print()
    
    # 3. Print statistics
    print("Dataset Statistics:")
    print(f"  Total rows: {report['total_rows']:,}")
    print(f"  Unique merchants: {report['statistics']['unique_merchants']:,}")
    print(f"  Amount range: ${report['statistics']['amount']['min']:.2f} - ${report['statistics']['amount']['max']:.2f}")
    print(f"  Mean amount: ${report['statistics']['amount']['mean']:.2f}")
    print(f"  Median amount: ${report['statistics']['amount']['median']:.2f}")
    if "date_range" in report["statistics"]:
        print(f"  Date range: {report['statistics']['date_range']['min']} to {report['statistics']['date_range']['max']}")
    print()
    
    print("Category Distribution:")
    for cat, count in sorted(report['statistics']['category_distribution'].items(), key=lambda x: -x[1]):
        pct = (count / report['total_rows']) * 100
        print(f"  {cat:20s}: {count:6,} ({pct:5.1f}%)")
    print()
    
    # 4. Save report
    print(f"Saving report to {REPORT_FILE}...")
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"✓ Report saved")
    print()
    
    # 5. Cleanup raw data
    cleanup_raw_data()
    print()
    
    # 6. Final summary
    print("="*60)
    if all_valid and report['total_rows'] >= 80000:
        print("✅ DATASET READY FOR TRAINING")
        print(f"   {report['total_rows']:,} clean rows with proper formatting")
        print()
        print("Files ready:")
        print(f"  • {MERGED_FILE}")
        print(f"  • {TAXONOMY_FILE}")
        print()
        print("Next steps:")
        print("  1. Seed database:")
        print("     PYTHONPATH=. python scripts/seed_database.py --csv data/merged_training.csv")
        print("  2. Build FAISS index:")
        print("     PYTHONPATH=. python scripts/build_faiss.py")
        print("  3. Train reranker:")
        print("     PYTHONPATH=. python scripts/train_reranker.py")
    else:
        print("⚠️  DATASET NEEDS ATTENTION")
        print(f"   Rows: {report['total_rows']:,}")
        print(f"   Valid: {all_valid}")
        if report["warnings"]:
            print("   Review warnings above")
    print("="*60)


if __name__ == "__main__":
    main()
