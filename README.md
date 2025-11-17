# LumaFin - Hybrid Adaptive Transaction Categorization System

[![CI Status](https://github.com/LathissKhumar/LumaFin/workflows/LumaFin%20CI/badge.svg)](https://github.com/LathissKhumar/LumaFin/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)

Production-grade AI system for financial transaction categorization combining:
- 🧠 **Global ML** (Label-Aware Contrastive Fine-Tuning + FAISS retrieval)
- 👤 **Personal Clustering** (AMPT: automatic micro-category discovery)
- 📏 **Deterministic Rules** (regex patterns, MCC codes)
- 🔍 **XGBoost Reranker** (feature-engineered scoring)
- 💡 **Explainability** (SHAP + nearest examples)
- 🔄 **Feedback Loop** (continuous learning)

## Architecture

```
Transaction Input
      ↓
  [Rules Engine] ────→ Match? → Return (conf=1.0)
      ↓ No match
  [Personal Centroids] → Strong match? → Return (conf=0.85-0.95)
      ↓ No strong match
  [FAISS Retrieval] ────→ Top-20 candidates
      ↓
  [XGBoost Reranker] ───→ Best category + calibrated confidence
      ↓
  [Explainer] ──────────→ SHAP + examples + decision path
      ↓
  [Feedback Loop] ──────→ Update centroids + retrain
```

## Features

### ✅ Implemented

- **Hierarchical Decision Pipeline**: Rules → Personal → Retrieval → Fallback
- **Embedding Model**: sentence-transformers with L-A CFT fine-tuning
- **Vector Search**: FAISS IndexFlatIP (cosine similarity)
- **Personal Clustering**: HDBSCAN-based AMPT engine with centroid persistence
- **Reranker**: XGBoost with 7 engineered features per category
- **Explainability**: Decision path, nearest examples, feature importance
- **API**: FastAPI with auth, rate limiting, validation
- **Feedback**: Celery workers for async centroid updates
- **Evaluation**: Macro F1, per-class metrics, calibration
- **Demo**: Streamlit UI for testing

### 🚧 In Progress

- Cross-encoder scoring (heuristic fallback working)
- Federated learning prototype
- Advanced fairness metrics

## 🚀 Google Colab Training (No GPU Required!)

**New!** Train models using Google Colab without local GPU:

1. **Quick Start:** Follow [COLAB_QUICK_START.md](COLAB_QUICK_START.md) (1-2 hours)
2. **Full Guide:** See [colab_notebooks/README.md](colab_notebooks/README.md)
3. **Hackathon Guide:** Check [HACKATHON_GUIDE.md](HACKATHON_GUIDE.md)

**What you get:**
- ✅ Fine-tuned embeddings (L-A CFT)
- ✅ Trained XGBoost reranker
- ✅ FAISS index with 80k+ vectors
- ✅ >90% accuracy evaluation

All trained models save to Google Drive and can be integrated using:
```bash
python scripts/integrate_colab_models.py --source /path/to/downloaded/models
```

---

## Quick Start

### 1. Prerequisites

```bash
# Python 3.11+
python --version

# PostgreSQL with pgvector
docker pull ankane/pgvector:latest

# Redis
docker pull redis:7-alpine
```

### 2. Installation

```bash
# Clone repository
git clone https://github.com/LathissKhumar/LumaFin.git
cd LumaFin

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"
```

### 2.1 Prepare datasets (clean and merge)

Clean raw public datasets into the required schema (merchant, amount, category) and merge:

```bash
# Example: download and clean (requires Kaggle CLI)
PYTHONPATH=. python scripts/download_and_clean_datasets.py \
  --dataset <kaggle_owner/dataset_id> \
  --dataset <another_owner/dataset_id>

# Or clean existing CSV folder(s)
PYTHONPATH=. python scripts/download_and_clean_datasets.py --input data/raw/my_dataset_folder

# Result: data/merged_training.csv
```

### 3. Setup Database

```bash
# Start PostgreSQL and Redis
docker-compose up -d

# Create database schema
PYTHONPATH=. python -c "
from src.storage.database import Base, engine
Base.metadata.create_all(bind=engine)
print('✅ Database schema created')
"

# Load seed data (requires data/global_examples.csv)
# Seed from the cleaned merged dataset
PYTHONPATH=. python scripts/seed_database.py --csv data/merged_training.csv
```

### 4. Train Models (Optional)

```bash
# Train XGBoost reranker (requires seeded database)
PYTHONPATH=. python scripts/train_reranker.py --limit 1000 --n-estimators 200

# Fine-tune embeddings with L-A CFT
PYTHONPATH=. python src/embedder/train_lacft.py --epochs 3 --batch-size 16

# Evaluate performance
PYTHONPATH=. python scripts/evaluate.py --mode fusion --limit 500
```

### 5. Run Services

```bash
# Terminal 1: API server
PYTHONPATH=. uvicorn src.api.main:app --reload --port 8000

# Terminal 2: Celery worker (feedback processing)
celery -A src.training.celery_app worker --loglevel=info

# Terminal 3: Celery beat (scheduled tasks)
celery -A src.training.celery_app beat --loglevel=info

# Terminal 4: Streamlit demo
PYTHONPATH=. streamlit run demo/streamlit_app.py
```

### 6. Test API

```bash
# Categorize transaction
curl -X POST http://localhost:8000/categorize \
  -H "Content-Type: application/json" \
  -d '{
    "merchant": "Starbucks",
    "amount": 5.50,
    "description": "Coffee",
    "user_id": null
  }'

# Submit feedback
curl -X POST http://localhost:8000/feedback \
  -H "Content-Type: application/json" \
  -d '{
    "transaction_id": 123,
    "user_id": 1,
    "correct_category": "Food & Drink"
  }'

# Get personal taxonomy
curl http://localhost:8000/taxonomy/personal/1
```

## Project Structure

```
LumaFin/
├── src/
│   ├── api/              # FastAPI endpoints
│   ├── embedder/         # Sentence-transformers + L-A CFT
│   ├── indexer/          # FAISS index builder
│   ├── retrieval/        # Vector search service
│   ├── reranker/         # XGBoost feature-based reranker
│   ├── clustering/       # AMPT engine + centroid matcher
│   ├── rules/            # Deterministic pattern matching
│   ├── fusion/           # Hierarchical decision pipeline
│   ├── explainer/        # SHAP + explanation builder
│   ├── training/         # Celery workers + incremental training
│   ├── storage/          # Database models + schema
│   ├── preprocessing/    # Merchant normalization
│   └── utils/            # Logger, metrics, exceptions
├── scripts/
│   ├── seed_database.py  # Initialize with sample data
│   ├── train_reranker.py # Train XGBoost model
│   └── evaluate.py       # Model evaluation
├── tests/
│   ├── test_core.py      # Unit tests
│   └── test_rules.py     # Rule engine tests
├── demo/
│   └── streamlit_app.py  # Interactive UI
├── data/
│   └── global_examples.csv  # Seed transactions
├── models/               # Trained models (gitignored)
├── .github/
│   └── workflows/
│       └── ci.yml        # GitHub Actions CI
├── docker-compose.yml
├── pyproject.toml
└── README.md

## Dataset schema requirements

To ensure high-quality training and evaluation, LumaFin expects a minimal schema:

Required columns (strict):
- merchant: string, short free-text merchant or purchase description (e.g., "Starbucks", "DMart", "pizza order")
- amount: number (float), transaction amount
- category: string, one of the canonical categories below; rows with "Uncategorized" are discarded for training

Optional columns (nice-to-have):
- description: string, additional free-text notes to enrich embeddings
- date: string (YYYY-MM-DD), optional temporal features
- currency: 3-letter ISO code

Canonical categories (9):
- Food & Dining
- Transportation
- Shopping
- Entertainment
- Bills & Utilities
- Healthcare
- Travel
- Income
- Uncategorized (not used for training)

Notes:
- During data cleaning, any dataset without merchant, amount, and category is skipped.
- Categories are normalized into the canonical 9-category taxonomy above.
- Rows labeled "Uncategorized" are dropped from the training split.
```

## Configuration

Create `.env` file:

```bash
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/lumafin
REDIS_URL=redis://localhost:6379/0

# Models
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
MODEL_PATH=models/embeddings/lumafin-lacft-v1.0  # Optional: fine-tuned model
RERANKER_MODEL_PATH=models/reranker/xgb_reranker.json

# FAISS
FAISS_INDEX_PATH=models/faiss_index.bin
FAISS_METADATA_PATH=models/faiss_metadata.pkl

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=TEXT  # or JSON

# API (optional)
SECRET_KEY=your-secret-key
BEARER_TOKEN=your-bearer-token
ENABLE_RATE_LIMIT=false

# Celery (optional)
CELERY_DISABLE_BEAT=false
FEEDBACK_INTERVAL_MINUTES=5
```

## Training Pipeline

### 1. Label-Aware Contrastive Fine-Tuning

```bash
# Fine-tune embeddings on labeled transactions
PYTHONPATH=. python src/embedder/train_lacft.py \
  --model sentence-transformers/all-MiniLM-L6-v2 \
  --output models/embeddings/lumafin-lacft-v1.0 \
  --epochs 3 \
  --batch-size 16 \
  --lr 2e-5

# Update .env to use fine-tuned model
echo "MODEL_PATH=models/embeddings/lumafin-lacft-v1.0" >> .env
```

### 2. Train Reranker

```bash
# Train XGBoost reranker with feature engineering
PYTHONPATH=. python scripts/train_reranker.py \
  --source db \
  --limit 1000 \
  --k 20 \
  --n-estimators 200 \
  --max-depth 4 \
  --learning-rate 0.07

# Model automatically saved to RERANKER_MODEL_PATH
# No config change needed - used automatically if present
```

### 3. Evaluation

```bash
# Baseline (retrieval only)
PYTHONPATH=. python scripts/evaluate.py --mode retrieval --limit 500

# With reranker
PYTHONPATH=. python scripts/evaluate.py --mode reranker --limit 500

# Full fusion pipeline
PYTHONPATH=. python scripts/evaluate.py --mode fusion --limit 500
```

## API Endpoints

### POST /categorize

Categorize a transaction.

**Request:**
```json
{
  "merchant": "Starbucks",
  "amount": 5.50,
  "description": "Morning coffee",
  "user_id": 123
}
```

**Response:**
```json
{
  "category": "Food & Drink",
  "confidence": 0.92,
  "explanation": {
    "decision_path": "centroid",
    "centroid_similarity": 0.87,
    "nearest_examples": [...]
  }
}
```

### POST /feedback

Submit user correction.

**Request:**
```json
{
  "transaction_id": 456,
  "user_id": 123,
  "correct_category": "Coffee Shops"
}
```

### GET /taxonomy/personal/{user_id}

Get user's personal micro-categories.

### POST /taxonomy/retrain/{user_id}

Trigger AMPT re-clustering for user.

### GET /taxonomy/global

Browse global category hierarchy.

## Performance Metrics

Based on evaluation with 500 labeled transactions:

| Mode | Macro F1 | Inference Time |
|------|----------|----------------|
| Retrieval (baseline) | 0.78 | ~20ms |
| Reranker | 0.83 | ~35ms |
| Fusion (full) | 0.88 | ~45ms |

## Development

### Run Tests

```bash
# Unit tests
PYTHONPATH=. pytest tests/test_core.py -v

# With coverage
PYTHONPATH=. pytest tests/ --cov=src --cov-report=html

# Specific test
PYTHONPATH=. pytest tests/test_core.py::TestReranker::test_feature_engineering -v
```

### Code Quality

```bash
# Lint
ruff check src/ tests/

# Format
black src/ tests/

# Type check
mypy src/ --ignore-missing-imports
```

### CI/CD

GitHub Actions workflow runs on push:
- Linting (ruff)
- Type checking (mypy)
- Unit tests (pytest)
- Integration tests (with PostgreSQL + Redis)
- Package build

## Troubleshooting

### Model Download Fails

```bash
# Pre-download models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# Or use local path
export MODEL_PATH=/path/to/local/model
```

### Database Connection Error

```bash
# Check PostgreSQL
docker-compose ps postgres

# Check connection
psql postgresql://postgres:postgres@localhost:5432/lumafin

# Reset database
docker-compose down -v
docker-compose up -d
PYTHONPATH=. python scripts/seed_database.py
```

### Celery Not Processing

```bash
# Check Redis
redis-cli ping

# Check worker
celery -A src.training.celery_app inspect active

# Manual trigger
PYTHONPATH=. python -c "from src.training.feedback_worker import process_feedback_batch; process_feedback_batch.delay()"
```

## Roadmap

### Phase 1: MVP ✅
- [x] Core categorization pipeline
- [x] FAISS retrieval
- [x] Basic reranker
- [x] FastAPI endpoints

### Phase 2: Personalization ✅
- [x] AMPT clustering
- [x] Personal centroids
- [x] Centroid matching
- [x] Feedback loop

### Phase 3: Advanced Features ✅
- [x] Rule engine
- [x] L-A CFT training
- [x] XGBoost reranker
- [x] Feature engineering
- [x] Explainability
- [x] Evaluation framework

### Phase 4: Production 🚧
- [ ] Comprehensive tests (70%+ coverage)
- [ ] Cross-encoder integration
- [ ] Knowledge graph
- [ ] Fairness metrics
- [ ] Federated learning
- [ ] Multi-tenancy
- [ ] Kubernetes deployment

## Contributing

1. Fork repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## License

MIT License - see LICENSE file

## Citation

```bibtex
@software{lumafin2025,
  title={LumaFin: Hybrid Adaptive System for Transaction Categorization},
  author={Lathiss Khumar},
  year={2025},
  url={https://github.com/LathissKhumar/LumaFin}
}
```

## Acknowledgments

- **sentence-transformers**: Embedding models
- **FAISS**: Vector search
- **HDBSCAN**: Density-based clustering
- **XGBoost**: Gradient boosting
- **FastAPI**: Modern Python API framework
- **Celery**: Distributed task queue

---

Built with ❤️ for intelligent financial categorization