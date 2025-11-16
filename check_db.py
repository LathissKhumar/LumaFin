import sys
sys.path.insert(0, '.')
from src.storage.database import SessionLocal
from sqlalchemy import text

db = SessionLocal()
result = db.execute(text("SELECT COUNT(*) FROM global_examples")).scalar()
print(f"Current row count: {result}")
db.close()
