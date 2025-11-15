# LumaFin Development Guide

## Quick Start Instructions

### 1. Seed the Database & Build FAISS Index

After setting up Docker and your virtual environment, run:

```bash
# Make sure Docker is running
docker-compose ps

# Activate virtual environment
source .venv/bin/activate

# Seed database with global examples and build FAISS index
PYTHONPATH=/home/lathiss/Projects/LumaFin python scripts/seed_database.py
```

This will:
- Load 100 example transactions from `data/global_examples.csv`
- Generate embeddings for each transaction
- Insert them into the `global_examples` table
- Build a FAISS index for fast similarity search
- Save the index to `models/faiss_index.bin`

### 2. Start the API Server

```bash
PYTHONPATH=/home/lathiss/Projects/LumaFin uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
```

### 3. Test the Categorization System

Open a new terminal and test:

```bash
# Health check
curl http://localhost:8000/health

# Test rule-based categorization (Netflix → Entertainment)
curl -X POST http://localhost:8000/categorize \
  -H "Content-Type: application/json" \
  -d '{"merchant": "Netflix", "amount": 15.99, "date": "2024-01-15"}'

# Test retrieval-based categorization (unknown merchant)
curl -X POST http://localhost:8000/categorize \
  -H "Content-Type: application/json" \
  -d '{"merchant": "Corner Cafe", "amount": 8.50, "date": "2024-01-15"}'

# View all loaded rules
curl http://localhost:8000/rules

# View global taxonomy
curl http://localhost:8000/taxonomy/global
```

### 4. View API Documentation

Visit in your browser:
- Interactive docs: http://localhost:8000/docs
- Alternative docs: http://localhost:8000/redoc

## System Architecture

### Hierarchical Decision Pipeline

The system categorizes transactions in this priority order:

1. **Rule Engine** (confidence = 1.0)
   - Regex pattern matching from database and YAML rules
   - Deterministic, always correct when matched
   - Example: "netflix" → Entertainment

2. **Personal Centroids** (confidence = 0.85-0.95) 
   - User-specific micro-categories from AMPT clustering
   - TODO: Implement in Phase 2

3. **Global Retrieval + Voting** (confidence = varies)
   - FAISS similarity search in global_examples
   - Vote aggregation from top-20 similar transactions
   - Returns category with highest vote share

4. **Fallback** (confidence = 0.0)
   - Returns "Uncategorized" if all above fail

### Current Implementation Status

✅ **Completed:**
- Project scaffold with modular structure
- Preprocessing (merchant normalization, amount bucketing)
- Pydantic models (Transaction, Category, Prediction, Explanation)
- Sentence-transformers embedder (384-dim vectors)
- Rule engine (DB + YAML rules, regex matching)
- FAISS indexer (IndexFlatIP with L2-normalized vectors)
- Retrieval service (similarity search + vote aggregation)
- FastAPI endpoints (/categorize, /health, /rules, /taxonomy/global)

🔨 **In Progress:**
- AMPT clustering engine (HDBSCAN, personal centroids)
- Cross-encoder reranker with XGBoost
- SHAP explainability

📋 **TODO:**
- Feedback loop (Celery workers)
- Streamlit demo UI
- Label-Aware Contrastive Fine-Tuning trainer
- Active learning query strategy
- Per-user evaluation metrics

## Development Workflow

### Running Tests

```bash
# Set PYTHONPATH for imports
export PYTHONPATH=/home/lathiss/Projects/LumaFin:$PYTHONPATH

# Run preprocessing tests
python tests/test_preprocessing.py

# Run rule engine tests
python tests/test_rules.py

# Or use pytest for all tests
pytest -v
```

### Adding New Rules

1. **Via Database:**
```sql
INSERT INTO rules (pattern, category_name, priority, is_active)
VALUES ('(?i)your_pattern', 'Category Name', 100, TRUE);
```

2. **Via YAML:** Edit `taxonomy/rules.yaml`
```yaml
rules:
  - pattern: '(?i)your_pattern'
    category: 'Category Name'
    priority: 100
```

3. **Refresh rules in API:**
```bash
curl -X POST http://localhost:8000/rules/refresh
```

### Adding New Global Examples

1. Add to `data/global_examples.csv`
2. Re-run seeding script:
```bash
PYTHONPATH=. python scripts/seed_database.py
```
3. Restart API to reload FAISS index

### Database Management

```bash
# Connect to PostgreSQL
docker exec -it lumafin_postgres psql -U lumafin -d lumafin_db

# View tables
\dt

# Check global examples
SELECT COUNT(*) FROM global_examples;

# Check rules
SELECT * FROM rules ORDER BY priority DESC;

# Exit
\q
```

## Troubleshooting

### Import Errors (ModuleNotFoundError)

Always set PYTHONPATH when running Python scripts:
```bash
export PYTHONPATH=/home/lathiss/Projects/LumaFin:$PYTHONPATH
```

Or prefix each command:
```bash
PYTHONPATH=/home/lathiss/Projects/LumaFin python your_script.py
```

### FAISS Index Not Found

If you get "No index built" errors:
1. Run the seed script: `PYTHONPATH=. python scripts/seed_database.py`
2. Verify files exist: `ls -lh models/faiss_*`
3. Restart API

### Database Connection Errors

```bash
# Check Docker containers
docker-compose ps

# View logs
docker-compose logs postgres

# Restart services
docker-compose restart
```

### SQLAlchemy Text Warnings

Always wrap raw SQL in `text()`:
```python
from sqlalchemy import text
db.execute(text("SELECT * FROM table"))
```

## Next Steps

### Phase 2: Personal Centroids (AMPT)

1. Implement `src/clustering/ampt_engine.py`
   - HDBSCAN clustering on user transaction history
   - Silhouette score evaluation
   - Cluster quality filtering

2. Implement `src/clustering/name_generator.py`
   - TF-IDF keyword extraction
   - Temporal pattern detection
   - Generate labels like "Morning Coffee" or "Weekend Groceries"

3. Implement `src/clustering/centroid_matcher.py`
   - Cosine similarity matching (threshold > 0.80)
   - Require ≥5 supporting transactions

4. Update `/categorize` endpoint to check centroids before retrieval

### Phase 3: Reranker & Explainability

1. Implement `src/reranker/model.py`
   - Cross-encoder for pairwise scoring
   - Feature engineering (FAISS scores, rule boosts, amount proximity)
   - XGBoost with Platt scaling

2. Implement `src/explainer/explain.py`
   - SHAP feature attributions
   - Natural language summaries
   - Counterfactual explanations

3. Add `/feedback` endpoint for user corrections

### Phase 4: Feedback Loop

1. Implement Celery workers for background processing
2. Build incremental training pipeline
3. Add active learning query selection
4. Deploy with Docker Compose orchestration

## API Reference

### POST /categorize

Categorize a transaction.

**Request:**
```json
{
  "merchant": "Starbucks",
  "amount": 5.50,
  "date": "2024-01-15",
  "description": "Coffee shop",
  "user_id": 1
}
```

**Response:**
```json
{
  "transaction": { ... },
  "category": {
    "name": "Food & Dining",
    "confidence": 1.0,
    "is_personal": false
  },
  "explanation": {
    "decision_path": "rule",
    "rule_matched": "Food & Dining",
    "nearest_examples": null
  },
  "timestamp": "2024-01-15T10:30:00"
}
```

### GET /health

Health check with system status.

**Response:**
```json
{
  "status": "ok",
  "database": "connected",
  "embedder_dimension": 384,
  "rules_loaded": 15
}
```

### GET /rules

List all loaded rules.

### GET /taxonomy/global

List all global categories.

### POST /rules/refresh

Reload rules from database and YAML.

## Project Structure

```
LumaFin/
├── src/
│   ├── api/              # FastAPI endpoints
│   ├── preprocessing/    # Merchant normalization
│   ├── embedder/         # Sentence-transformers wrapper
│   ├── indexer/          # FAISS index builder
│   ├── retrieval/        # Similarity search service
│   ├── rules/            # Pattern matching engine
│   ├── clustering/       # AMPT engine (TODO)
│   ├── reranker/         # Cross-encoder (TODO)
│   ├── explainer/        # SHAP explanations (TODO)
│   ├── fusion/           # Decision combiner (TODO)
│   ├── training/         # Feedback workers (TODO)
│   ├── storage/          # Database connection
│   └── models.py         # Pydantic schemas
├── data/
│   └── global_examples.csv  # Seed transaction data
├── models/               # Saved FAISS index & models
├── taxonomy/
│   └── rules.yaml        # Default categorization rules
├── scripts/
│   └── seed_database.py  # Database initialization
├── tests/                # Unit & integration tests
├── demo/                 # Streamlit UI (TODO)
├── docker-compose.yml    # PostgreSQL + Redis
├── .env                  # Configuration
└── README.md             # This file
```

## Contributing

When adding new features:
1. Follow the hierarchical decision pipeline
2. Add unit tests in `tests/`
3. Update this README
4. Use type hints and docstrings
5. Run linting: `black src/ && ruff src/`

## License

[Your License Here]
