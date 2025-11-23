"""
Script to seed database with global examples and build FAISS index.

Run this once after initializing the database.
Supports specifying an alternative CSV file containing columns
merchant, amount, category (case-insensitive variants accepted).
"""
import csv
import sys
import os
import argparse

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from sqlalchemy import text
from src.storage.database import SessionLocal
from src.embedder.encoder import TransactionEmbedder
from src.indexer.faiss_builder import FAISSIndexBuilder
import json
from pathlib import Path


def seed_global_examples(csv_path: str = 'data/global_examples.csv'):
    """Load examples from CSV and insert into database with embeddings."""
    db = SessionLocal()
    embedder = TransactionEmbedder()

    try:
        print(f"Loading global examples from CSV: {csv_path}")
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            cols = {c.lower(): c for c in (reader.fieldnames or [])}
            merch_col = cols.get('merchant') or cols.get('text') or cols.get('description') or cols.get('name')
            amt_col = cols.get('amount') or cols.get('amt') or cols.get('value') or cols.get('price')
            cat_col = cols.get('category') or cols.get('label')
            if not merch_col or not cat_col:
                raise RuntimeError("CSV must contain merchant/text and category/label columns")
            examples = []
            for row in reader:
                merchant = row.get(merch_col, '').strip()
                if not merchant:
                    continue
                amount_val = 0.0
                if amt_col and row.get(amt_col):
                    try:
                        amount_val = float(row.get(amt_col))
                    except Exception:
                        amount_val = 0.0
                category_val = row.get(cat_col, 'Uncategorized') or 'Uncategorized'
                examples.append({'merchant': merchant, 'amount': amount_val, 'category': category_val})

        print(f"Found {len(examples)} examples")

        # Load canonical taxonomy from data/taxonomy.json and ensure only
        # canonical categories are used for seeding (do NOT add arbitrary CSV categories).
        taxonomy_path = Path('data/taxonomy.json')
        if not taxonomy_path.exists():
            raise RuntimeError("taxonomy.json not found in data/ - required to normalize categories")

        taxonomy = json.loads(taxonomy_path.read_text())
        canonical_names = [c['name'] for c in taxonomy.get('categories', [])]

        # Ensure canonical categories exist in DB (insert only these, not CSV-derived categories)
        result = db.execute(text("SELECT id, category_name FROM global_taxonomy")).fetchall()
        category_map = {row[1]: row[0] for row in result}
        for cname in canonical_names:
            if cname not in category_map:
                db.execute(text("INSERT INTO global_taxonomy (category_name, parent_category, description, level) VALUES (:name, NULL, '', 1) ON CONFLICT (category_name) DO NOTHING"), {'name': cname})
        db.commit()
        # reload mapping
        result = db.execute(text("SELECT id, category_name FROM global_taxonomy")).fetchall()
        category_map = {row[1]: row[0] for row in result}

        # Build keyword mapping for normalization
        keywords_map = {}
        for c in taxonomy.get('categories', []):
            kws = set([k.lower() for k in c.get('keywords', [])])
            # include the category name itself and simple tokens
            kws.add(c['name'].lower())
            for token in c['name'].lower().replace('&', ' ').split():
                kws.add(token.strip())
            keywords_map[c['name']] = kws

        def normalize_category(raw_cat: str, merchant: str) -> str:
            """Map raw CSV category to canonical taxonomy using MERCHANT NAME as primary signal.
            Since 97.6% of CSV has 'Uncategorized', we must use merchant text to predict category.
            Returns canonical name (string).
            """
            if not raw_cat:
                raw_cat = 'Uncategorized'
            
            merchant_lower = merchant.lower().strip()
            raw_lower = raw_cat.strip().lower()
            
            # === STEP 1: If CSV category is already good (not "uncategorized"), try to map it ===
            if raw_lower != 'uncategorized':
                # Exact match to canonical
                for cname in canonical_names:
                    if raw_lower == cname.lower():
                        return cname
                
                # Explicit CSV→Canonical mappings
                explicit_maps = {
                    # → Food & Dining
                'food': 'Food & Dining',
                
                # → Shopping (includes apparel, household, gifts, etc.)
                'apparel': 'Shopping',
                'household': 'Shopping',
                'gift': 'Shopping',
                'beauty': 'Shopping',
                
                # → Entertainment
                'culture': 'Entertainment',
                'festivals': 'Entertainment',
                
                # → Bills & Utilities
                'subscription': 'Bills & Utilities',
                'maid': 'Bills & Utilities',
                
                # → Healthcare
                'health': 'Healthcare',
                
                # → Travel
                'tourism': 'Travel',
                
                # → Income (savings, refunds, dividends, interest, investments)
                'dividend earned on shares': 'Income',
                'interest': 'Income',
                'salary': 'Income',
                'saving bank account 1': 'Income',
                'refund': 'Income',
                'income': 'Income',
                
                # → Uncategorized (ambiguous)
                'other': 'Uncategorized',
                'family': 'Uncategorized',
            }
            
            # Check explicit mappings first (most reliable)
            if s in explicit_maps:
                return explicit_maps[s]
            
            # === KEYWORD-BASED MAPPINGS (more flexible) ===
            # Check keywords from taxonomy
            for cname, kws in keywords_map.items():
                for kw in kws:
                    if not kw or len(kw) < 2:
                        continue
                    if kw in s:
                        return cname
            
            # === COMPREHENSIVE HEURISTIC MAPPINGS ===
            # Food & Dining
            if any(term in s for term in ['food', 'dining', 'restaurant', 'grocery', 'cafe', 'coffee', 
                                          'breakfast', 'lunch', 'dinner', 'snack', 'meal', 'eat', 'bhaji',
                                          'vadapav', 'pav', 'chai', 'pizza', 'poha', 'atta', 'bread',
                                          'milk', 'dahi', 'butter', 'kachori', 'samosa', 'vada', 'idli']):
                return 'Food & Dining'
            
            # Transportation
            if any(term in s for term in ['transport', 'taxi', 'uber', 'lyft', 'train', 'bus', 'metro',
                                          'gas', 'fuel', 'petrol', 'parking', 'toll', 'vehicle', 'car', 'bike',
                                          'ola', 'cab', 'express', 'railway', 'station']):
                return 'Transportation'
            
            # Shopping (apparel, household, gifts, etc.)
            if any(term in s for term in ['shop', 'apparel', 'clothing', 'clothes', 'fashion', 'retail',
                                          'household', 'furniture', 'gift', 'present', 'beauty', 'grooming',
                                          'cosmetic', 'personal care', 'accessories', 'ironing', 'supermart',
                                          'decathlon', 'myntra', 'amazon', 'chappal', 'shoes', 'dress',
                                          'towel', 'shampoo', 'water purifier', 'bulb', 'pencil', 'mouse',
                                          'cover', 'purse', 'chappal', 'slides', 'earphones', 'radio',
                                          'umbrella', 'repair']):
                return 'Shopping'
            
            # Entertainment
            if any(term in s for term in ['entertainment', 'movie', 'cinema', 'netflix', 'hulu', 'spotify',
                                          'disney', 'gaming', 'game', 'music', 'streaming', 'concert',
                                          'theatre', 'culture', 'festival', 'hobby', 'recreation', 'inox',
                                          'pvr', 'planetarium', 'cinema', 'theatre']):
                return 'Entertainment'
            
            # Bills & Utilities
            if any(term in s for term in ['bill', 'utility', 'rent', 'electric', 'water', 'gas',
                                          'internet', 'phone', 'mobile', 'subscription', 'membership',
                                          'service', 'insurance', 'maid', 'cleaning', 'recharge', 'tata play',
                                          'wifi', 'router']):
                return 'Bills & Utilities'
            
            # Healthcare
            if any(term in s for term in ['health', 'medical', 'doctor', 'hospital', 'clinic', 'pharmacy',
                                          'medicine', 'drug', 'prescription', 'dental', 'therapy', 'wellness',
                                          'vaccine', 'covaxin', 'cataract', 'eye', 'glucose', 'strepsils',
                                          'tablet', 'injection', 'consultation']):
                return 'Healthcare'
            
            # Travel
            if any(term in s for term in ['travel', 'hotel', 'flight', 'airbnb', 'vacation', 'trip',
                                          'tour', 'tourism', 'booking', 'airline', 'resort', 'park inn']):
                return 'Travel'
            
            # Income (salary, refunds, dividends, interest, investments)
            if any(term in s for term in ['income', 'salary', 'wage', 'payroll', 'bonus', 'reward',
                                          'refund', 'deposit', 'transfer', 'dividend', 'interest',
                                          'investment', 'maturity', 'mutual fund', 'stock', 'equity',
                                          'recurring deposit', 'fixed deposit', 'ppf', 'provident', 'fd',
                                          'rd', 'savings', 'atm', 'gpay', 'workplace']):
                return 'Income'
            
            # Default: map to Uncategorized
            return 'Uncategorized'

        # Prepare data for insertion — normalize CSV categories to canonical ones
        prepared = []
        for ex in examples:
            merchant = ex.get('merchant', '')
            category = ex.get('category', 'Uncategorized')
            norm = normalize_category(category, merchant)
            # Only keep examples that map to one of the canonical taxonomy names
            if norm not in category_map:
                # skip if canonical category somehow missing (shouldn't happen)
                continue
            cat_id = category_map[norm]
            prepared.append({
                'merchant': merchant,
                'amount': float(ex.get('amount', 0.0)),
                'category': norm,
                'category_id': int(cat_id),
                'description': ex.get('description', '')
            })

        print(f"Prepared {len(prepared)} examples for insertion")

        # Batch encode and insert
        print("Encoding transactions in batches...")
        batch_size = int(os.getenv('SEED_BATCH_SIZE', '256'))
        inserted = 0

        for i in range(0, len(prepared), batch_size):
            batch = prepared[i:i+batch_size]
            # Use embedder.encode_batch which returns numpy array
            embeddings = embedder.encode_batch(batch)

            # Build params for executemany
            params = []
            for j, item in enumerate(batch):
                emb_list = embeddings[j].astype(float).tolist()
                params.append({
                    'merchant': item['merchant'],
                    'amount': item['amount'],
                    'category_id': item['category_id'],
                    'embedding': json.dumps(emb_list),
                    'description': item.get('description', '')
                })

            insert_sql = text("""
                INSERT INTO global_examples (merchant, amount, category_id, embedding, description)
                VALUES (:merchant, :amount, :category_id, :embedding, :description)
            """)
            db.execute(insert_sql, params)
            db.commit()
            inserted += len(params)
            print(f"  → Inserted {inserted}/{len(prepared)}")

        print(f"✓ Inserted {inserted} examples into database")

    except Exception as e:
        db.rollback()
        print(f"✗ Error: {e}")
        raise
    finally:
        db.close()


def build_faiss_index():
    """Build FAISS index from database examples."""
    db = SessionLocal()
    builder = FAISSIndexBuilder()

    try:
        print("\nBuilding FAISS index...")
        
        # Load examples from DB
        examples, embeddings = builder.load_examples_from_db(db)
        print(f"Loaded {len(examples)} examples from database")

        # Build index
        builder.build_index(embeddings, examples)
        print(f"✓ Built FAISS index with {len(embeddings)} vectors")

        # Save to disk
        os.makedirs('models', exist_ok=True)
        builder.save('models/faiss_index.bin', 'models/faiss_metadata.pkl')

        # Test search
        print("\nTesting index with sample query...")
        test_embedding = builder.embedder.encode_transaction("Starbucks", 5.50)
        results = builder.search(test_embedding, k=3)
        
        print("\nTop 3 similar transactions for 'Starbucks $5.50':")
        for i, result in enumerate(results, 1):
            print(f"  {i}. {result['merchant']} (${result['amount']:.2f}) "
                  f"-> {result['category']} (similarity: {result['similarity']:.3f})")

    except Exception as e:
        print(f"✗ Error: {e}")
        raise
    finally:
        db.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', default='data/global_examples.csv', help='Path to CSV containing global examples')
    args = parser.parse_args()

    print("=== LumaFin Database Seeding ===\n")

    # Seed examples
    seed_global_examples(args.csv)

    # Build index
    build_faiss_index()

    print("\n✓ Database seeding complete!")
    print("\nNext steps:")
    print("  1. Start API: PYTHONPATH=. uvicorn src.api.main:app --reload")
    print("  2. Test categorization: curl -X POST http://localhost:8000/categorize ...")
