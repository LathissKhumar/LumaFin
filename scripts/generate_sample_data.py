"""Generate sample transaction data for testing and demo.

Creates realistic synthetic transactions across multiple categories.

Usage:
  python scripts/generate_sample_data.py --output data/sample_transactions.csv --count 1000
"""
from __future__ import annotations

import argparse
import csv
import random
from datetime import datetime, timedelta
from typing import List, Dict

# Sample merchants by category
MERCHANTS = {
    "Food & Dining": [
        "Starbucks", "McDonald's", "Chipotle", "Panera Bread", "Subway",
        "Domino's Pizza", "Taco Bell", "Olive Garden", "The Cheesecake Factory",
        "Local Cafe", "Food Truck", "Restaurant"
    ],
    "Groceries": [
        "Whole Foods", "Trader Joe's", "Safeway", "Costco", "Target",
        "Walmart", "Kroger", "Aldi", "Local Grocery", "Farmers Market"
    ],
    "Transportation": [
        "Uber", "Lyft", "Shell Gas", "Chevron", "BP Gas Station",
        "Metro Transit", "Parking Meter", "Car Wash", "Auto Repair"
    ],
    "Entertainment": [
        "Netflix", "Spotify", "Amazon Prime", "Disney+", "HBO Max",
        "Movie Theater", "Concert Venue", "Bowling Alley", "Arcade"
    ],
    "Shopping": [
        "Amazon", "Target", "Best Buy", "Home Depot", "IKEA",
        "Macy's", "Nike Store", "Apple Store", "Online Shop"
    ],
    "Healthcare": [
        "CVS Pharmacy", "Walgreens", "Medical Center", "Dental Office",
        "Eye Doctor", "Physical Therapy", "Urgent Care"
    ],
    "Utilities": [
        "Electric Company", "Water Utility", "Internet Provider", "Gas Company",
        "Phone Bill", "Trash Service"
    ],
    "Personal Care": [
        "Haircut Salon", "Gym Membership", "Spa", "Nail Salon", "Barbershop"
    ],
    "Travel": [
        "Delta Airlines", "United Airlines", "Marriott Hotel", "Hilton",
        "Airbnb", "Booking.com", "Travel Agency"
    ],
    "Bills & Fees": [
        "Bank Fee", "ATM Withdrawal", "Credit Card Payment", "Late Fee",
        "Subscription Service"
    ]
}

# Amount ranges by category (min, max)
AMOUNT_RANGES = {
    "Food & Dining": (3.0, 60.0),
    "Groceries": (15.0, 200.0),
    "Transportation": (5.0, 80.0),
    "Entertainment": (8.0, 50.0),
    "Shopping": (10.0, 500.0),
    "Healthcare": (20.0, 300.0),
    "Utilities": (40.0, 150.0),
    "Personal Care": (15.0, 100.0),
    "Travel": (50.0, 1500.0),
    "Bills & Fees": (5.0, 100.0)
}


def generate_transaction(
    category: str,
    start_date: datetime,
    end_date: datetime
) -> Dict[str, any]:
    """Generate a single realistic transaction."""
    merchants = MERCHANTS[category]
    amount_min, amount_max = AMOUNT_RANGES[category]
    
    # Random merchant from category
    merchant = random.choice(merchants)
    
    # Random amount within range
    amount = round(random.uniform(amount_min, amount_max), 2)
    
    # Random date
    time_delta = end_date - start_date
    random_days = random.randint(0, time_delta.days)
    random_hours = random.randint(0, 23)
    random_minutes = random.randint(0, 59)
    transaction_date = start_date + timedelta(
        days=random_days,
        hours=random_hours,
        minutes=random_minutes
    )
    
    # Optional description
    descriptions = [
        None,
        f"Purchase at {merchant}",
        f"Payment",
        f"Online order",
        f"In-store purchase"
    ]
    description = random.choice(descriptions) if random.random() > 0.7 else None
    
    return {
        "merchant": merchant,
        "amount": amount,
        "category": category,
        "date": transaction_date.isoformat(),
        "description": description
    }


def generate_dataset(
    count: int,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    category_weights: Dict[str, float] | None = None
) -> List[Dict[str, any]]:
    """Generate a dataset of synthetic transactions.
    
    Args:
        count: Number of transactions to generate
        start_date: Start date for transactions (default: 90 days ago)
        end_date: End date for transactions (default: today)
        category_weights: Optional dict of category -> probability
    
    Returns:
        List of transaction dicts
    """
    if start_date is None:
        start_date = datetime.now() - timedelta(days=90)
    if end_date is None:
        end_date = datetime.now()
    
    # Default equal weights
    categories = list(MERCHANTS.keys())
    if category_weights is None:
        category_weights = {cat: 1.0 / len(categories) for cat in categories}
    
    transactions = []
    
    for _ in range(count):
        # Select category based on weights
        category = random.choices(
            categories,
            weights=[category_weights.get(cat, 0.1) for cat in categories]
        )[0]
        
        txn = generate_transaction(category, start_date, end_date)
        transactions.append(txn)
    
    return transactions


def save_to_csv(transactions: List[Dict[str, any]], output_path: str):
    """Save transactions to CSV file."""
    fieldnames = ["merchant", "amount", "category", "date", "description"]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(transactions)
    
    print(f"✅ Saved {len(transactions)} transactions to {output_path}")


def print_summary(transactions: List[Dict[str, any]]):
    """Print dataset summary statistics."""
    from collections import Counter
    
    category_counts = Counter(txn["category"] for txn in transactions)
    total_amount = sum(txn["amount"] for txn in transactions)
    
    print("\n📊 Dataset Summary")
    print(f"Total transactions: {len(transactions)}")
    print(f"Total amount: ${total_amount:,.2f}")
    print(f"Average amount: ${total_amount / len(transactions):.2f}")
    print(f"\nTransactions by category:")
    for category, count in category_counts.most_common():
        pct = 100 * count / len(transactions)
        print(f"  {category:20s}: {count:4d} ({pct:5.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="Generate sample transaction data")
    parser.add_argument(
        "--output",
        default="data/sample_transactions.csv",
        help="Output CSV file path"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of transactions to generate"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=90,
        help="Number of days of history to generate"
    )
    args = parser.parse_args()
    
    print(f"🎲 Generating {args.count} sample transactions...")
    
    # Generate with more frequent categories
    weights = {
        "Food & Dining": 0.25,
        "Groceries": 0.20,
        "Shopping": 0.15,
        "Transportation": 0.12,
        "Entertainment": 0.10,
        "Healthcare": 0.05,
        "Utilities": 0.05,
        "Personal Care": 0.03,
        "Travel": 0.03,
        "Bills & Fees": 0.02
    }
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=args.days)
    
    transactions = generate_dataset(
        count=args.count,
        start_date=start_date,
        end_date=end_date,
        category_weights=weights
    )
    
    # Save to CSV
    save_to_csv(transactions, args.output)
    
    # Print summary
    print_summary(transactions)
    
    print(f"\n✅ Sample data generation complete!")
    print(f"Use this data to:")
    print(f"  1. Seed database: PYTHONPATH=. python scripts/seed_database.py")
    print(f"  2. Train models: PYTHONPATH=. python scripts/train_reranker.py")
    print(f"  3. Evaluate: PYTHONPATH=. python scripts/evaluate.py")


if __name__ == "__main__":
    main()
