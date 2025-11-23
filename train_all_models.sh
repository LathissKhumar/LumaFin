#!/bin/bash
# Complete training pipeline for LumaFin with 81k dataset

set -e  # Exit on error

echo "=================================="
echo "LUMAFIN COMPLETE TRAINING PIPELINE"
echo "Dataset: 81,001 rows"
echo "=================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Function to print status
print_status() {
    echo -e "${GREEN}[$(date +'%H:%M:%S')]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

# Activate virtual environment if it exists
if [ -d ".venv" ]; then
    print_status "Activating virtual environment..."
    source .venv/bin/activate
fi

# Set environment
export PYTHONPATH=.
export TOKENIZERS_PARALLELISM=false
export SEED_BATCH_SIZE=256

# Check if database is running
print_status "Checking database connection..."
if ! python -c "from src.storage.database import SessionLocal; db = SessionLocal(); db.close()" 2>/dev/null; then
    print_warning "Database not accessible, starting docker-compose..."
    docker-compose up -d
    print_status "Waiting 20 seconds for database to initialize..."
    sleep 20
    if ! python -c "from src.storage.database import SessionLocal; db = SessionLocal(); db.close()" 2>/dev/null; then
        print_error "Database still not accessible. Checking logs..."
        docker logs lumafin_postgres --tail 10
        exit 1
    fi
fi
print_status "✓ Database connection OK"
echo ""

# Step 1: Seed Database
print_status "STEP 1/4: Seeding database with 81k transactions..."
print_status "This will take 10-15 minutes for embedding generation..."
if python scripts/seed_database.py --csv data/merged_training.csv 2>&1 | tee /tmp/seed_log.txt; then
    print_status "✓ Database seeded successfully"
else
    print_error "Database seeding failed. Check /tmp/seed_log.txt"
    tail -20 /tmp/seed_log.txt
    exit 1
fi
echo ""

# Step 2: Build FAISS Index
print_status "STEP 2/4: Building FAISS index for fast retrieval..."
if python scripts/build_faiss.py 2>&1 | tee /tmp/faiss_log.txt; then
    print_status "✓ FAISS index built successfully"
else
    print_error "FAISS index build failed. Check /tmp/faiss_log.txt"
    tail -20 /tmp/faiss_log.txt
    exit 1
fi
echo ""

# Step 3: Train Reranker
print_status "STEP 3/4: Training XGBoost reranker..."
print_status "Using 2000 examples with retrieval candidates..."
if python scripts/train_reranker.py --source db --limit 2000 --k 20 --n-estimators 200 --max-depth 4 2>&1 | tee /tmp/reranker_log.txt; then
    print_status "✓ Reranker trained successfully"
else
    print_error "Reranker training failed. Check /tmp/reranker_log.txt"
    tail -20 /tmp/reranker_log.txt
    exit 1
fi
echo ""

# Step 4: Train L-A CFT Embeddings (Optional but recommended)
print_status "STEP 4/4: Training Label-Aware Contrastive Fine-Tuned embeddings..."
print_status "This is optional but improves accuracy by 5-8%"
print_status "Training with 3 epochs on CSV data..."
if python src/embedder/train_lacft.py \
    --csv data/merged_training.csv \
    --epochs 3 \
    --batch-size 32 \
    --lr 2e-5 \
    --output-dir models/embeddings/lumafin-lacft-v1.0 \
    --balance \
    --hard-negatives 2>&1 | tee /tmp/lacft_log.txt; then
    print_status "✓ L-A CFT embeddings trained successfully"
    print_status "To use fine-tuned model, set: MODEL_PATH=models/embeddings/lumafin-lacft-v1.0"
else
    print_warning "L-A CFT training failed (optional step). System will use base model."
    print_warning "Check /tmp/lacft_log.txt for details"
fi
echo ""

# Final Summary
echo "=================================="
echo "TRAINING PIPELINE COMPLETE!"
echo "=================================="
echo ""
echo "Models trained:"
echo "  ✓ Global embeddings (81k examples)"
echo "  ✓ FAISS index (fast retrieval)"
echo "  ✓ XGBoost reranker (calibrated)"
if [ -d "models/embeddings/lumafin-lacft-v1.0" ]; then
    echo "  ✓ L-A CFT fine-tuned embeddings"
fi
echo ""
echo "Next steps:"
echo "  1. Start API server:"
echo "     uvicorn src.api.main:app --reload"
echo ""
echo "  2. Run evaluation:"
echo "     python scripts/evaluate.py --mode fusion --limit 1000"
echo ""
echo "  3. Start Streamlit demo:"
echo "     streamlit run demo/streamlit_app.py"
echo ""
echo "Logs saved to /tmp/*_log.txt"
echo "=================================="
