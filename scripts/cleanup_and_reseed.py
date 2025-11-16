"""Clean up database and reseed with proper category normalization."""
import sys
sys.path.insert(0, '.')

from sqlalchemy import text
from src.storage.database import SessionLocal

db = SessionLocal()
try:
    print("Cleaning up database...")
    
    # Delete all examples and non-canonical categories
    db.execute(text("DELETE FROM global_examples"))
    db.execute(text("DELETE FROM global_taxonomy WHERE category_name NOT IN ('Food & Dining', 'Transportation', 'Shopping', 'Entertainment', 'Bills & Utilities', 'Healthcare', 'Travel', 'Income', 'Uncategorized')"))
    db.commit()
    
    print("✓ Database cleaned")
    print("\nNow run: PYTHONPATH=. python scripts/seed_database.py --csv data/merged_training.csv")
    
except Exception as e:
    db.rollback()
    print(f"Error: {e}")
finally:
    db.close()
