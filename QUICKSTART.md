# LumaFin Quick Start Guide

Get LumaFin running in **under 10 minutes** with this streamlined guide.

## 🚀 1-Minute Setup (Docker Compose)

```bash
# Clone and enter directory
git clone https://github.com/LathissKhumar/LumaFin.git
cd LumaFin

# Start services (PostgreSQL + Redis)
docker-compose up -d

# Install Python dependencies
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e .

# Generate sample data
python scripts/generate_sample_data.py --count 500 --output data/global_examples.csv

# Initialize database
PYTHONPATH=. python -c "from src.storage.database import Base, engine; Base.metadata.create_all(bind=engine)"

# Load data and build index
PYTHONPATH=. python scripts/seed_database.py
```

**✅ You're ready! Start the API:**

```bash
PYTHONPATH=. uvicorn src.api.main:app --reload
```

Visit: http://localhost:8000/docs

## 📊 Test Categorization

```bash
curl -X POST http://localhost:8000/categorize \
  -H "Content-Type: application/json" \
  -d '{
    "merchant": "Starbucks",
    "amount": 5.50
  }'
```

Expected response:
```json
{
  "category": "Food & Dining",
  "confidence": 0.87,
  "explanation": {
    "decision_path": "retrieval",
    "nearest_examples": [...]
  }
}
```

## 🎯 Optional: Train Models

### Train XGBoost Reranker (2-3 minutes)

```bash
PYTHONPATH=. python scripts/train_reranker.py --limit 500
```

This improves accuracy by 5-10%.

### Fine-tune Embeddings (10-15 minutes, requires GPU for speed)

```bash
PYTHONPATH=. python src/embedder/train_lacft.py --epochs 3
```

This optimizes embeddings for financial transactions.

## 🔄 Enable Feedback Loop

```bash
# Terminal 1: Celery worker
celery -A src.training.celery_app worker --loglevel=info

# Terminal 2: Celery beat (scheduler)
celery -A src.training.celery_app beat --loglevel=info
```

Now user corrections automatically update personal centroids!

## 🎨 Launch Demo UI

```bash
PYTHONPATH=. streamlit run demo/streamlit_app.py
```

Visit: http://localhost:8501

## 📈 Evaluate Performance

```bash
# Baseline (retrieval only)
PYTHONPATH=. python scripts/evaluate.py --mode retrieval

# With reranker
PYTHONPATH=. python scripts/evaluate.py --mode reranker

# Full pipeline
PYTHONPATH=. python scripts/evaluate.py --mode fusion
```

## 🧪 Run Tests

```bash
PYTHONPATH=. pytest tests/test_core.py -v
```

## 🐛 Troubleshooting

### "Module not found" errors
```bash
export PYTHONPATH=$(pwd)
```

### Database connection fails
```bash
docker-compose ps  # Check services are running
docker-compose logs postgres
```

### Models download slowly
```bash
# Pre-download models
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

### Port 8000 already in use
```bash
# Use different port
uvicorn src.api.main:app --port 8001
```

## 📚 Next Steps

1. **Read the full README**: `README.md`
2. **Deploy to production**: `DEPLOYMENT.md`
3. **Add custom rules**: Edit `taxonomy/rules.yaml`
4. **Customize categories**: Update `data/taxonomy.json`
5. **Scale with Kubernetes**: See `DEPLOYMENT.md`

## 💡 Common Use Cases

### Personal Finance App
- User submits transactions
- System categorizes automatically
- User corrects mistakes
- Personal patterns emerge over time

### Business Expense Tracking
- Employees upload receipts
- Automatic categorization by department
- Enforce spending policies with rules
- Export for accounting

### Banking Integration
- Ingest transaction feeds
- Real-time categorization
- Custom user taxonomies
- Fraud detection signals

## 🎓 Learning Resources

- **Plan Document**: `.github/prompts/plan-lumaFinHybridAdaptiveSystem.prompt.md`
- **API Documentation**: http://localhost:8000/docs (when running)
- **Code Examples**: `tests/test_core.py`
- **Architecture**: See README.md

## 📞 Need Help?

- **Issues**: https://github.com/LathissKhumar/LumaFin/issues
- **Discussions**: https://github.com/LathissKhumar/LumaFin/discussions

---

**🎉 Enjoy using LumaFin!**
