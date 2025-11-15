"""
Script to seed database with global examples and build FAISS index.

Run this once after initializing the database.
"""
import csv
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

from sqlalchemy import text
from src.storage.database import SessionLocal
from src.embedder.encoder import TransactionEmbedder
from src.indexer.faiss_builder import FAISSIndexBuilder


def seed_global_examples():
    """Load examples from CSV and insert into database with embeddings."""
    db = SessionLocal()
    embedder = TransactionEmbedder()

    try:
        print("Loading global examples from CSV...")
        with open('data/global_examples.csv', 'r') as f:
            reader = csv.DictReader(f)
            examples = list(reader)

        print(f"Found {len(examples)} examples")

        # Get category IDs
        result = db.execute(text("SELECT id, category_name FROM global_taxonomy"))
        category_map = {row[1]: row[0] for row in result}

        inserted = 0
        for example in examples:
            merchant = example['merchant']
            amount = float(example['amount'])
            category = example['category']

            # Get category ID
            category_id = category_map.get(category)
            if not category_id:
                print(f"Warning: Unknown category '{category}' for {merchant}")
                continue

            # Generate embedding
            embedding = embedder.encode_transaction(merchant, amount)
            embedding_list = embedding.tolist()

            # Insert
            db.execute(text("""
                INSERT INTO global_examples (merchant, amount, category_id, embedding)
                VALUES (:merchant, :amount, :category_id, :embedding)
            """), {
                'merchant': merchant,
                'amount': amount,
                'category_id': category_id,
                'embedding': embedding_list
            })
            inserted += 1

        db.commit()
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
    print("=== LumaFin Database Seeding ===\n")
    
    # Seed examples
    seed_global_examples()
    
    # Build index
    build_faiss_index()
    
    print("\n✓ Database seeding complete!")
    print("\nNext steps:")
    print("  1. Start API: PYTHONPATH=. uvicorn src.api.main:app --reload")
    print("  2. Test categorization: curl -X POST http://localhost:8000/categorize ...")
