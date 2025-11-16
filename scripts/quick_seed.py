#!/usr/bin/env python3
"""Quick seeding script to load cleaned data into database."""

import sys
sys.path.insert(0, '.')

import csv
from pathlib import Path
from sqlalchemy import text
from src.storage.database import SessionLocal
from src.embedder.encoder import TransactionEmbedder
from src.indexer.faiss_builder import FAISSIndexBuilder

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data/merged_training.csv"

print(f"Loading data from {CSV_PATH}...")
db = SessionLocal()
embedder = TransactionEmbedder()

try:
    # 1. Clean old data
    print("Cleaning old data...")
    db.execute(text("DELETE FROM global_examples"))
    db.execute(text("DELETE FROM global_taxonomy WHERE category_name NOT IN ('Food & Dining', 'Transportation', 'Shopping', 'Entertainment', 'Bills & Utilities', 'Healthcare', 'Travel', 'Income', 'Uncategorized')"))
    db.commit()
    
    # 2. Ensure canonical categories exist
    print("Ensuring canonical categories...")
    canonical = ['Food & Dining', 'Transportation', 'Shopping', 'Entertainment', 
                 'Bills & Utilities', 'Healthcare', 'Travel', 'Income', 'Uncategorized']
    for cat in canonical:
        db.execute(text("""
            INSERT INTO global_taxonomy (category_name, parent_category, description, level)
            VALUES (:name, NULL, '', 1)
            ON CONFLICT (category_name) DO NOTHING
        """), {'name': cat})
    db.commit()
    
    # Get category ID map
    result = db.execute(text("SELECT id, category_name FROM global_taxonomy")).fetchall()
    category_map = {row[1]: row[0] for row in result}
    print(f"Category map: {category_map}")
    
    # 3. Load CSV and insert
    print(f"Loading {CSV_PATH}...")
    examples = []
    with open(CSV_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            merchant = row.get('merchant', '').strip()
            amount = float(row.get('amount', 0.0))
            category = row.get('category', 'Uncategorized').strip()
            
            if not merchant or category not in category_map:
                continue
                
            examples.append({
                'merchant': merchant,
                'amount': amount,
                'category': category,
                'category_id': category_map[category]
            })
    
    print(f"Inserting {len(examples)} examples...")
    for ex in examples:
        # Generate embedding
        emb = embedder.encode(ex['merchant'])
        
        # Insert into DB
        db.execute(text("""
            INSERT INTO global_examples (merchant, amount, category_id, embedding)
            VALUES (:merchant, :amount, :category_id, :embedding)
        """), {
            'merchant': ex['merchant'],
            'amount': ex['amount'],
            'category_id': ex['category_id'],
            'embedding': emb.tolist()
        })
    
    db.commit()
    print(f"✓ Inserted {len(examples)} examples")
    
    # 4. Build FAISS index
    print("Building FAISS index...")
    builder = FAISSIndexBuilder()
    builder.build_index()
    print("✓ FAISS index built")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
    db.rollback()
finally:
    db.close()
