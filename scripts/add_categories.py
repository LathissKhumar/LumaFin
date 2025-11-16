"""Add all necessary categories to the database."""
import sys
sys.path.insert(0, '.')

from sqlalchemy import text
from src.storage.database import SessionLocal

# All categories needed for the merged dataset
categories = [
    # Core categories (already in schema)
    ('Food & Dining', None, 'Restaurants, groceries, cafes', 1),
    ('Transportation', None, 'Gas, parking, public transit', 1),
    ('Shopping', None, 'Retail purchases', 1),
    ('Entertainment', None, 'Movies, games, hobbies', 1),
    ('Bills & Utilities', None, 'Rent, electricity, internet', 1),
    ('Healthcare', None, 'Medical, pharmacy, insurance', 1),
    ('Travel', None, 'Hotels, flights, vacation', 1),
    ('Income', None, 'Salary, refunds, transfers in', 1),
    ('Uncategorized', None, 'Unknown or unclassified', 1),
    
    # Additional categories from dataset
    ('Other', None, 'Miscellaneous expenses', 1),
    ('Food', None, 'Food and beverages', 1),  # Alias for Food & Dining
    ('Apparel', None, 'Clothing and accessories', 1),
    ('Household', None, 'Household items and supplies', 1),
    ('Festivals', None, 'Festival-related expenses', 1),
    ('Beauty', None, 'Beauty and grooming', 1),
    ('Grooming', None, 'Personal grooming services', 1),
    ('Health', None, 'Health and medical expenses', 1),
    ('subscription', None, 'Subscriptions and memberships', 1),
    ('Salary', 'Income', 'Employment income', 2),
    ('Dividend earned on Shares', 'Income', 'Investment dividends', 2),
    ('Interest', 'Income', 'Interest earned or paid', 2),
    ('Gift', None, 'Gifts given or received', 1),
    ('Education', None, 'Education and learning expenses', 1),
    ('Family', None, 'Family-related expenses', 1),
    ('Culture', None, 'Cultural activities', 1),
    ('Tourism', None, 'Travel and tourism', 1),
    ('Investment', None, 'Investment transactions', 1),
    ('Bonus', 'Income', 'Bonuses and rewards', 2),
    ('Equity Mutual Fund A', 'Investment', 'Mutual fund investments', 2),
    ('Equity Mutual Fund B', 'Investment', 'Mutual fund investments', 2),
    ('Equity Mutual Fund C', 'Investment', 'Mutual fund investments', 2),
    ('Equity Mutual Fund D', 'Investment', 'Mutual fund investments', 2),
    ('Equity Mutual Fund E', 'Investment', 'Mutual fund investments', 2),
    ('Equity Mutual Fund F', 'Investment', 'Mutual fund investments', 2),
    ('Fixed Deposit', 'Investment', 'Fixed deposit investments', 2),
    ('Recurring Deposit', 'Investment', 'Recurring deposits', 2),
    ('Public Provident Fund', 'Investment', 'PPF investments', 2),
    ('Share Market', 'Investment', 'Stock market investments', 2),
    ('Maturity amount', 'Income', 'Investment maturity proceeds', 2),
    ('Gpay Reward', 'Income', 'Digital payment rewards', 2),
    ('water (jar /tanker)', 'Bills & Utilities', 'Water supply', 2),
    ('maid', 'Household', 'Domestic help', 2),
    ('Money transfer', None, 'Money transfers', 1),
    ('Petty cash', None, 'Small cash expenses', 1),
]

db = SessionLocal()
try:
    print("Adding categories to database...")
    added = 0
    for cat_name, parent, desc, level in categories:
        try:
            db.execute(text("""
                INSERT INTO global_taxonomy (category_name, parent_category, description, level)
                VALUES (:name, :parent, :desc, :level)
                ON CONFLICT (category_name) DO NOTHING
            """), {'name': cat_name, 'parent': parent, 'desc': desc, 'level': level})
            added += 1
        except Exception as e:
            print(f"Warning: Could not add '{cat_name}': {e}")
    
    db.commit()
    print(f"✓ Added {added} categories")
    
    # Verify
    result = db.execute(text("SELECT COUNT(*) FROM global_taxonomy"))
    count = result.scalar()
    print(f"✓ Total categories in database: {count}")
    
except Exception as e:
    db.rollback()
    print(f"✗ Error: {e}")
    raise
finally:
    db.close()
