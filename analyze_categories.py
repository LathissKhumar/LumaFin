#!/usr/bin/env python3
import csv
from collections import Counter

with open('data/merged_training.csv', 'r') as f:
    reader = csv.DictReader(f)
    categories = [row['category'] for row in reader if row.get('category')]

print(f"Total rows: {len(categories)}")
print(f"\nUnique categories and their counts:\n")
counts = Counter(categories)
for cat, count in counts.most_common():
    pct = (count / len(categories)) * 100
    print(f"{cat:40} : {count:6} ({pct:5.2f}%)")
