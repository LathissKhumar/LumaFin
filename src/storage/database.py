import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

engine = create_engine(DATABASE_URL, echo=False)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

def get_db():
    """Dependency for FastAPI endpoints"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def sample_negative_examples(db, excluded_category_name: str | None, limit: int = 5, max_attempts: int = 10):
    """Sample negative examples efficiently without using ORDER BY RANDOM().

    This uses random id lookups (indexed) to fetch rows from `global_examples`
    that do not match the excluded category. It avoids full table random scans.
    Returns list of rows with columns: id, merchant, amount, description, category_name, embedding
    """
    import random
    from sqlalchemy import text

    # Get max id for simple random seeding
    max_id = db.execute(text("SELECT COALESCE(MAX(id), 1) FROM global_examples")).scalar() or 1
    collected = []
    attempts = 0
    while len(collected) < limit and attempts < max_attempts:
        attempts += 1
        rand_id = random.randint(1, max(1, int(max_id)))
        # Try to retrieve some rows with id >= rand_id
        q = text(
            "SELECT ge.id, ge.merchant, ge.amount, ge.description, gt.category_name, ge.embedding "
            "FROM global_examples ge JOIN global_taxonomy gt ON ge.category_id = gt.id "
            "WHERE (:label IS NULL OR gt.category_name != :label) AND ge.id >= :r LIMIT :lim"
        )
        rows = db.execute(q, {"label": excluded_category_name, "r": rand_id, "lim": limit - len(collected)}).fetchall()
        for r in rows:
            if r not in collected:
                collected.append(r)
        # If not enough yet, fetch rows with id < rand_id
        if len(collected) < limit:
            q2 = text(
                "SELECT ge.id, ge.merchant, ge.amount, ge.description, gt.category_name, ge.embedding "
                "FROM global_examples ge JOIN global_taxonomy gt ON ge.category_id = gt.id "
                "WHERE (:label IS NULL OR gt.category_name != :label) AND ge.id < :r LIMIT :lim"
            )
            rows2 = db.execute(q2, {"label": excluded_category_name, "r": rand_id, "lim": limit - len(collected)}).fetchall()
            for r in rows2:
                if r not in collected:
                    collected.append(r)
    return collected[:limit]


def sample_random_examples(db, limit: int = 100, max_attempts: int = 10):
    """Sample random examples from the `global_examples` table without ORDER BY RANDOM().

    Returns rows similar to sample_negative_examples but without excluding categories.
    """
    return sample_negative_examples(db, None, limit=limit, max_attempts=max_attempts)