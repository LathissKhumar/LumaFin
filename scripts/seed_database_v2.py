"""
Script to seed database with global examples using MERCHANT-BASED weak labeling.

Since 97.6% of CSV data is labeled "Uncategorized", we use merchant names 
to predict categories with high-precision rules.
"""
import csv
import sys
import os
import argparse
import json
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from sqlalchemy import text
from src.storage.database import SessionLocal
from src.embedder.encoder import TransactionEmbedder
from src.indexer.faiss_builder import FAISSIndexBuilder


def predict_category_from_merchant(merchant: str, raw_category: str, canonical_names: list) -> str:
    """Predict category using merchant name (primary) and CSV label (fallback).
    
    Returns: canonical category name
    """
    merchant_lower = merchant.lower().strip()
    raw_lower = raw_category.lower().strip() if raw_category else 'uncategorized'
    
    # === INCOME (very distinct - check first) ===
    if any(term in merchant_lower for term in [
        'salary', 'wage', 'payroll', 'bonus', 'workplace', 'employer',
        'dividend', 'interest', 'mutual fund', 'stock', 'equity', 'share',
        'fixed deposit', 'fd', 'recurring deposit', 'rd', 'ppf', 'provident',
        'maturity', 'investment', 'refund', 'cashback', 'reward', 'rebate',
        'reimbursement', 'gpay reward', 'return', 'saving', 'deposit'
    ]):
        return 'Income'
    
    # === FOOD & DINING ===
    # NOTE: Dataset has DESCRIPTIVE transactions like "idli medu vada mix 2 plates" not merchant names
    if any(term in merchant_lower for term in [
        # Food items (Indian)
        'idli', 'vada', 'medu', 'dosa', 'bhaji', 'vadapav', 'pav', 'chai', 'tea', 'poha',
        'biryani', 'dahi', 'paneer', 'roti', 'naan', 'curry', 'masala', 'samosa', 'kachori',
        'paratha', 'thali', 'rice', 'dal', 'sabzi', 'raita', 'curd',
        # Food items (Global)
        'pizza', 'burger', 'sandwich', 'pasta', 'noodle', 'chicken', 'fish', 'egg',
        'plate', 'plates',  # "2 plates", "mix 2 plates"
        # Groceries
        'atta', 'flour', 'bread', 'milk', 'butter', 'cheese', 'yogurt', 'oil', 'ghee',
        'sugar', 'salt', 'spice', 'vegetable', 'fruit', 'onion', 'potato', 'tomato',
        'mango', 'apple', 'banana', 'orange', 'grape',
        # Restaurants & Cafes
        'restaurant', 'cafe', 'coffee', 'starbucks', 'mcdonald', 'kfc', 'domino', 'subway',
        'food', 'dining', 'kitchen', 'eatery', 'bistro', 'diner', 'canteen', 'mess',
        # Delivery & Groceries
        'grocery', 'supermarket', 'mart', 'bakery', 'sweets', 'snack', 'meal',
        'breakfast', 'lunch', 'dinner', 'eat', 'zomato', 'swiggy', 'uber eats'
    ]):
        return 'Food & Dining'
    
    # === TRANSPORTATION ===
    if any(term in merchant_lower for term in [
        # Ride services
            'ola', 'uber', 'lyft', 'cab', 'taxi', 'rapido', 'meru', 'auto',
            # Public transport & Locations (dataset has "place 2 to place 3" patterns)
            'railway', 'train', 'metro', 'bus', 'irctc', 'redbus', 'station',
            'place to', 'to place', 'residence to', 'to residence',  # descriptive travel
        # Fuel
            'petrol', 'diesel', 'gas', 'fuel', 'pump', 'cng', 'shell', 'bp', 'essar', 'indian oil',
        # Other
            'parking', 'toll', 'fastag', 'vehicle', 'car', 'bike', 'scooter',
            'transport', 'commute', 'travel', 'ride'
    ]):
        return 'Transportation'
    
    # === HEALTHCARE ===
    if any(term in merchant_lower for term in [
        # Facilities
        'hospital', 'clinic', 'doctor', 'medical', 'pharmacy', 'drugstore', 'medicine',
        'apollo', 'fortis', 'max', 'care', 'medic', 'health', 'dental', 'dentist',
        # Medical
        'vaccine', 'covaxin', 'covishield', 'injection', 'tablet', 'pill', 'capsule',
        'cataract', 'surgery', 'eye', 'glucose', 'diabetes', 'bp', 'blood',
        'consultation', 'checkup', 'lab', 'test', 'xray', 'scan', 'mri', 'ct',
        'strepsils', 'cough', 'cold', 'fever', 'diagnosis'
    ]):
        return 'Healthcare'
    
    # === BILLS & UTILITIES ===
    if any(term in merchant_lower for term in [
        # Utilities
        'electricity', 'electric', 'power', 'utility', 'water', 'sewage', 'gas utility',
        # Telecom
        'internet', 'broadband', 'wifi', 'airtel', 'jio', 'vodafone', 'bsnl', 'idea',
            'mobile', 'phone', 'recharge', 'telecom', 'sim', 'data',
        # Subscriptions
            'subscription', 'month subscription', 'membership', 'netflix', 'prime', 'hotstar', 'spotify',
        'youtube premium', 'disney', 'hulu',
        # Rent & Services
        'rent', 'lease', 'maid', 'housekeeper', 'cleaning', 'maintenance',
        'society', 'hoa', 'tata play', 'dish', 'cable'
    ]):
        return 'Bills & Utilities'
    
    # === SHOPPING ===
    if any(term in merchant_lower for term in [
        # Online
        'amazon', 'flipkart', 'myntra', 'ajio', 'nykaa', 'paytm', 'snapdeal',
        # Retail
        'shop', 'shopping', 'store', 'retail', 'mall', 'market', 'bazaar', 'supermart',
        'decathlon', 'reliance', 'dmart', 'big bazaar', 'more', 'spencer',
        # Apparel & Footwear
        'clothes', 'clothing', 'apparel', 'fashion', 'dress', 'shirt', 'pant', 'jeans',
        'shoe', 'shoes', 'chappal', 'sandal', 'footwear', 'slipper', 'slides',
        'accessories', 'wear', 'tshirt', 't-shirt', 'kurti', 'saree', 'salwar',
        # Household items (look for specific items in descriptions)
        'household', 'furniture', 'appliance', 'electronics', 'gadget', 'device',
        'towel', 'bedsheet', 'curtain', 'pillow', 'mattress', 'blanket',
        'utensil', 'crockery', 'plate', 'cup', 'glass', 'spoon', 'fork', 'knife',
        'fan', 'iron', 'cooker', 'mixer', 'grinder', 'bulb', 'light', 'torch',
        # Personal care & Beauty
        'gift', 'present', 'beauty', 'cosmetic', 'makeup', 'grooming',
        'shampoo', 'soap', 'detergent', 'cream', 'lotion', 'powder', 'perfume',
        # Accessories & Misc
        'purse', 'wallet', 'bag', 'backpack', 'umbrella', 'watch', 'jewel', 'jewelry',
        'ring', 'earring', 'necklace', 'chain', 'bracelet',
        'book', 'stationery', 'pen', 'pencil', 'notebook', 'paper',
        'toy', 'game board', 'doll', 'teddy'
    ]):
        return 'Shopping'
    
    # === ENTERTAINMENT ===
    if any(term in merchant_lower for term in [
        # Movies
        'movie', 'cinema', 'pvr', 'inox', 'theatre', 'multiplex', 'cinepolis',
        # Gaming
        'entertainment', 'game', 'gaming', 'playstation', 'xbox', 'nintendo', 'steam',
        # Events
            'concert', 'show', 'festival', 'ganesh', 'idol', 'puja', 'event', 'ticket', 'bookmyshow',
        'museum', 'planetarium', 'zoo', 'park', 'amusement',
        # Social
        'hobby', 'recreation', 'club', 'bar', 'pub', 'lounge', 'nightclub'
    ]):
        return 'Entertainment'
    
    # === TRAVEL ===
    if any(term in merchant_lower for term in [
        # Accommodation
        'hotel', 'resort', 'lodge', 'inn', 'accommodation', 'motel',
        # Airlines
        'flight', 'airline', 'airport', 'indigo', 'spicejet', 'air india', 'vistara',
        # Booking platforms
        'makemytrip', 'goibibo', 'yatra', 'cleartrip', 'booking.com', 'expedia',
        'airbnb', 'oyo', 'treebo', 'fab hotels', 'zostel',
        # General
        'tour', 'tourism', 'vacation', 'holiday', 'trip', 'package', 'cruise'
    ]):
        return 'Travel'
    
    # === FALLBACK: Use CSV label if it's not "uncategorized" ===
    if raw_lower != 'uncategorized':
        explicit_maps = {
            'food': 'Food & Dining',
            'transportation': 'Transportation',
            'apparel': 'Shopping',
            'household': 'Shopping',
            'gift': 'Shopping',
            'beauty': 'Shopping',
            'grooming': 'Shopping',
            'education': 'Shopping',
            'self-development': 'Shopping',
            'documents': 'Shopping',
            'culture': 'Entertainment',
            'festivals': 'Entertainment',
            'social life': 'Entertainment',
            'subscription': 'Bills & Utilities',
            'maid': 'Bills & Utilities',
            'water (jar /tanker)': 'Bills & Utilities',
            'health': 'Healthcare',
            'tourism': 'Travel',
            'dividend earned on shares': 'Income',
            'interest': 'Income',
            'salary': 'Income',
            'bonus': 'Income',
            'investment': 'Income',
            'saving bank account 1': 'Income',
            'maturity amount': 'Income',
            'money transfer': 'Income',
            'fixed deposit': 'Income',
            'recurring deposit': 'Income',
            'public provident fund': 'Income',
            'equity mutual fund a': 'Income',
            'equity mutual fund d': 'Income',
            'equity mutual fund e': 'Income',
            'equity mutual fund f': 'Income',
            'share market': 'Income',
            'petty cash': 'Uncategorized',
            'other': 'Uncategorized',
            'family': 'Uncategorized',
        }
        
        if raw_lower in explicit_maps:
            return explicit_maps[raw_lower]
    
    # Final fallback
    return 'Uncategorized'


def seed_global_examples(csv_path: str = 'data/merged_training.csv'):
    """Load examples from CSV, predict categories from merchant names, and insert with embeddings."""
    db = SessionLocal()
    embedder = TransactionEmbedder()

    try:
        print(f"\n=== LumaFin Database Seeding ===\n")
        print(f"Loading global examples from CSV: {csv_path}")
        
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            cols = {c.lower(): c for c in (reader.fieldnames or [])}
            merch_col = cols.get('merchant') or cols.get('text') or cols.get('description')
            amt_col = cols.get('amount') or cols.get('amt')
            cat_col = cols.get('category') or cols.get('label')
            
            if not merch_col or not cat_col:
                raise RuntimeError("CSV must contain merchant and category columns")
            
            examples = []
            for row in reader:
                merchant = row.get(merch_col, '').strip()
                if not merchant:
                    continue
                
                amount_val = 0.0
                if amt_col and row.get(amt_col):
                    try:
                        amount_val = float(row.get(amt_col))
                    except:
                        amount_val = 0.0
                
                category_val = row.get(cat_col, 'Uncategorized') or 'Uncategorized'
                examples.append({
                    'merchant': merchant,
                    'amount': amount_val,
                    'csv_category': category_val
                })

        print(f"Found {len(examples)} examples")

        # Load canonical taxonomy
        taxonomy_path = Path('data/taxonomy.json')
        if not taxonomy_path.exists():
            raise RuntimeError("taxonomy.json not found")

        taxonomy = json.loads(taxonomy_path.read_text())
        canonical_names = [c['name'] for c in taxonomy.get('categories', [])]

        # Ensure canonical categories exist in DB
        result = db.execute(text("SELECT id, category_name FROM global_taxonomy")).fetchall()
        category_map = {row[1]: row[0] for row in result}
        
        for cname in canonical_names:
            if cname not in category_map:
                db.execute(text("INSERT INTO global_taxonomy (category_name, parent_category, description, level) VALUES (:name, NULL, '', 1) ON CONFLICT (category_name) DO NOTHING"), {'name': cname})
        db.commit()
        
        result = db.execute(text("SELECT id, category_name FROM global_taxonomy")).fetchall()
        category_map = {row[1]: row[0] for row in result}

        # Predict categories using merchant names
        print(f"Predicting categories from merchant names...")
        prepared = []
        for ex in examples:
            predicted_category = predict_category_from_merchant(
                ex['merchant'],
                ex['csv_category'],
                canonical_names
            )
            
            if predicted_category not in category_map:
                continue
            
            prepared.append({
                'merchant': ex['merchant'],
                'amount': float(ex['amount']),
                'category': predicted_category,
                'category_id': int(category_map[predicted_category])
            })

        print(f"Prepared {len(prepared)} examples for insertion")

        # Show category distribution BEFORE inserting
        from collections import Counter
        cat_counts = Counter([p['category'] for p in prepared])
        print(f"\n📊 PREDICTED CATEGORY DISTRIBUTION:")
        print("=" * 60)
        for cat, count in sorted(cat_counts.items(), key=lambda x: -x[1]):
            pct = 100.0 * count / len(prepared)
            print(f"  {cat:25} │ {count:6,} │ {pct:6.2f}%")
        
        unc_pct = 100.0 * cat_counts.get('Uncategorized', 0) / len(prepared)
        proper_pct = 100 - unc_pct
        print("=" * 60)
        print(f"  Uncategorized: {unc_pct:.2f}%")
        print(f"  Properly Categorized: {proper_pct:.2f}%")
        if proper_pct >= 90:
            print(f"  ✅ TARGET ACHIEVED: {proper_pct:.2f}% ≥ 90%")
        else:
            print(f"  ⚠️  Gap to target: {90 - proper_pct:.2f}%")
        print("=" * 60)

        # Batch encode and insert
        print("\nEncoding transactions in batches...")
        batch_size = int(os.getenv('SEED_BATCH_SIZE', '256'))
        inserted = 0

        for i in range(0, len(prepared), batch_size):
            batch = prepared[i:i+batch_size]
            embeddings = embedder.encode_batch(batch)

            params = []
            for j, item in enumerate(batch):
                emb_list = embeddings[j].astype(float).tolist()
                params.append({
                    'merchant': item['merchant'],
                    'amount': item['amount'],
                    'category_id': item['category_id'],
                    'embedding': json.dumps(emb_list)
                })

            insert_sql = text("""
                INSERT INTO global_examples (merchant, amount, category_id, embedding)
                VALUES (:merchant, :amount, :category_id, CAST(:embedding AS vector))
            """)
            
            db.execute(insert_sql, params)
            db.commit()
            
            inserted += len(batch)
            print(f"  → Inserted {inserted}/{len(prepared)}")

        print(f"✓ Inserted {inserted} examples into database\n")

    except Exception as e:
        db.rollback()
        print(f"Error during seeding: {e}")
        raise
    finally:
        db.close()


def build_faiss_index():
    """Build FAISS index from database examples."""
    print("Building FAISS index...")
    builder = FAISSIndexBuilder()
    builder.build_from_database()
    print("✓ Built FAISS index with vectors")
    print(f"✓ Saved index to models/faiss_index.bin")
    print(f"✓ Saved metadata to models/faiss_metadata.pkl\n")

    # Test query
    print("Testing index with sample query...\n")
    from src.retrieval.service import RetrievalService
    retrieval = RetrievalService()
    results = retrieval.search("Starbucks $5.50", k=3)
    
    print("Top 3 similar transactions for 'Starbucks $5.50':")
    for i, r in enumerate(results, 1):
        print(f"  {i}. {r['merchant']} (${r['amount']:.2f}) -> {r['category']} (similarity: {r['similarity']:.3f})")
    
    print(f"\n✓ Database seeding complete!\n")
    print("Next steps:")
    print("  1. Start API: PYTHONPATH=. uvicorn src.api.main:app --reload")
    print("  2. Test categorization: curl -X POST http://localhost:8000/categorize ...\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Seed database with global examples')
    parser.add_argument('--csv', default='data/merged_training.csv', help='Path to CSV file')
    args = parser.parse_args()

    seed_global_examples(args.csv)
    build_faiss_index()
