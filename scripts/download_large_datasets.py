"""
Download and prepare large-scale transaction datasets from multiple sources.

This script downloads datasets from:
1. Kaggle: Credit card transactions dataset (if kaggle.json configured)
2. Synthetic transaction generator (fallback)
3. UCI ML Repository datasets

Target: 80,000+ clean training examples
"""
import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import random
import requests
from typing import List, Dict

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

OUTPUT_DIR = Path("data/raw")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def download_kaggle_dataset(dataset_name: str, output_path: Path) -> bool:
    """Download dataset from Kaggle using kaggle API."""
    try:
        import kaggle
        print(f"Downloading Kaggle dataset: {dataset_name}")
        kaggle.api.dataset_download_files(
            dataset_name,
            path=str(output_path),
            unzip=True
        )
        print(f"✓ Downloaded {dataset_name}")
        return True
    except Exception as e:
        print(f"✗ Failed to download {dataset_name}: {e}")
        return False


def generate_synthetic_transactions(n: int = 50000) -> pd.DataFrame:
    """Generate synthetic but realistic transaction data."""
    print(f"Generating {n} synthetic transactions...")
    
    # Merchant categories with realistic names
    merchants = {
        "Food & Dining": [
            "Starbucks", "McDonald's", "Subway", "Chipotle", "Panera Bread",
            "Whole Foods", "Trader Joe's", "Safeway", "Kroger", "Walmart Grocery",
            "Costco Food Court", "Pizza Hut", "Domino's Pizza", "KFC", "Burger King",
            "Taco Bell", "Wendy's", "Chick-fil-A", "Five Guys", "In-N-Out Burger",
            "Local Cafe", "Diner", "Sushi Restaurant", "Italian Restaurant", "Chinese Takeout"
        ],
        "Shopping": [
            "Amazon", "Target", "Walmart", "Best Buy", "Home Depot",
            "Lowe's", "Costco", "Sam's Club", "Macy's", "Nordstrom",
            "Gap", "H&M", "Zara", "Nike Store", "Apple Store",
            "Office Depot", "Staples", "CVS", "Walgreens", "TJ Maxx"
        ],
        "Transportation": [
            "Uber", "Lyft", "Shell Gas Station", "Chevron", "BP Gas",
            "Exxon", "Mobil", "Metro Transit", "Parking Meter", "Toll Road",
            "Airport Parking", "Car Wash", "Auto Repair Shop", "Tesla Supercharger"
        ],
        "Entertainment": [
            "Netflix", "Spotify", "Hulu", "Disney+", "HBO Max",
            "AMC Theaters", "Regal Cinemas", "Steam", "PlayStation Store", "Xbox Store",
            "Concert Tickets", "Sports Stadium", "Museum", "Theme Park"
        ],
        "Bills & Utilities": [
            "Electric Company", "Gas Utility", "Water Bill", "Internet Provider",
            "Cell Phone Bill", "Cable TV", "Streaming Bundle", "HOA Fees",
            "Gym Membership", "Insurance Payment", "Rent Payment"
        ],
        "Healthcare": [
            "CVS Pharmacy", "Walgreens Pharmacy", "Doctor's Office", "Dentist",
            "Hospital", "Urgent Care", "Lab Test", "Vision Center", "Physical Therapy"
        ],
        "Travel": [
            "Airbnb", "Hotel Booking", "Airlines", "Expedia", "Booking.com",
            "Car Rental", "Vacation Resort", "Train Tickets", "Bus Tickets"
        ],
        "Income": [
            "Salary Deposit", "Direct Deposit", "Paycheck", "Bonus", "Tax Refund",
            "Dividend", "Interest", "Cashback Reward", "Refund"
        ]
    }
    
    # Generate transactions
    data = []
    start_date = datetime.now() - timedelta(days=730)  # 2 years of history
    
    for _ in range(n):
        # Pick random category and merchant
        category = random.choice(list(merchants.keys()))
        merchant = random.choice(merchants[category])
        
        # Generate realistic amounts based on category
        if category == "Food & Dining":
            amount = round(random.gauss(25, 15), 2)
            amount = max(3.0, min(amount, 200.0))
        elif category == "Shopping":
            amount = round(random.gauss(75, 50), 2)
            amount = max(10.0, min(amount, 500.0))
        elif category == "Transportation":
            amount = round(random.gauss(35, 20), 2)
            amount = max(5.0, min(amount, 150.0))
        elif category == "Entertainment":
            amount = round(random.gauss(15, 10), 2)
            amount = max(5.0, min(amount, 100.0))
        elif category == "Bills & Utilities":
            amount = round(random.gauss(100, 50), 2)
            amount = max(20.0, min(amount, 300.0))
        elif category == "Healthcare":
            amount = round(random.gauss(150, 100), 2)
            amount = max(20.0, min(amount, 1000.0))
        elif category == "Travel":
            amount = round(random.gauss(300, 200), 2)
            amount = max(50.0, min(amount, 2000.0))
        else:  # Income
            amount = round(random.gauss(2000, 1000), 2)
            amount = max(100.0, min(amount, 10000.0))
        
        # Generate date
        days_ago = random.randint(0, 730)
        date = start_date + timedelta(days=days_ago)
        
        # Add some variation to merchant names
        if random.random() < 0.2:
            merchant = f"{merchant} #{random.randint(100, 999)}"
        
        data.append({
            "merchant": merchant,
            "amount": amount,
            "category": category,
            "date": date.strftime("%Y-%m-%d"),
            "description": f"Purchase at {merchant}"
        })
    
    df = pd.DataFrame(data)
    print(f"✓ Generated {len(df)} synthetic transactions")
    return df


def download_github_datasets() -> List[pd.DataFrame]:
    """Download transaction datasets from public GitHub repositories."""
    datasets = []
    
    # Example: Try to download sample datasets from public repos
    urls = [
        # Add any public transaction datasets you find
        # "https://raw.githubusercontent.com/example/dataset.csv"
    ]
    
    for url in urls:
        try:
            print(f"Downloading from {url}...")
            df = pd.read_csv(url)
            datasets.append(df)
            print(f"✓ Downloaded {len(df)} rows")
        except Exception as e:
            print(f"✗ Failed: {e}")
    
    return datasets


def main():
    print("="*60)
    print("LUMAFIN DATASET ACQUISITION")
    print("="*60)
    print()
    
    all_datasets = []
    
    # 1. Try Kaggle datasets
    print("Step 1: Attempting Kaggle downloads...")
    kaggle_datasets = [
        "kartik2112/fraud-detection",  # Has transaction data
        # Add more dataset IDs here
    ]
    
    for ds in kaggle_datasets:
        kaggle_dir = OUTPUT_DIR / "kaggle"
        if download_kaggle_dataset(ds, kaggle_dir):
            # Try to find CSV files in downloaded data
            for csv_file in kaggle_dir.rglob("*.csv"):
                try:
                    df = pd.read_csv(csv_file)
                    if len(df) > 100:  # Only keep substantial datasets
                        all_datasets.append(("kaggle", csv_file.name, df))
                        print(f"  → Loaded {csv_file.name}: {len(df)} rows")
                except Exception as e:
                    print(f"  ✗ Could not load {csv_file.name}: {e}")
    
    print()
    
    # 2. Generate synthetic data
    print("Step 2: Generating synthetic transactions...")
    synthetic_df = generate_synthetic_transactions(50000)
    all_datasets.append(("synthetic", "generated", synthetic_df))
    print()
    
    # 3. Try GitHub datasets
    print("Step 3: Attempting GitHub downloads...")
    github_dfs = download_github_datasets()
    for i, df in enumerate(github_dfs):
        all_datasets.append(("github", f"dataset_{i}", df))
    print()
    
    # 4. Save all raw datasets
    print("Step 4: Saving raw datasets...")
    for source, name, df in all_datasets:
        safe_name = name.replace("/", "_").replace("\\", "_")
        output_path = OUTPUT_DIR / f"{source}_{safe_name}.csv"
        df.to_csv(output_path, index=False)
        print(f"✓ Saved {output_path}: {len(df)} rows")
    
    print()
    print("="*60)
    print(f"SUMMARY: Downloaded {len(all_datasets)} datasets")
    total_rows = sum(len(df) for _, _, df in all_datasets)
    print(f"Total rows: {total_rows:,}")
    print()
    print("Next step: Run scripts/clean_and_merge_datasets.py")
    print("="*60)


if __name__ == "__main__":
    main()
