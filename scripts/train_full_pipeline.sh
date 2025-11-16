#!/usr/bin/env bash
# Full training pipeline orchestrator for LumaFin
# Runs all steps from data prep to evaluation

set -e  # Exit on error

echo "=== LumaFin Full Training Pipeline ==="
echo

# Step 1: Prepare merged training CSV
echo "[1/6] Preparing merged training data from Kaggle CSVs..."
PYTHONPATH=. python scripts/prepare_kaggle_data.py \
  --input data/kaggle \
  --output data/merged_training.csv
echo "✓ Merged training CSV ready"
echo

# Step 2: Fine-tune L-A CFT embedder
echo "[2/6] Fine-tuning Label-Aware Contrastive embedder..."
PYTHONPATH=. python src/embedder/train_lacft.py \
  --epochs 3 \
  --batch-size 32 \
  --csv data/merged_training.csv \
  --hard-negatives \
  --balance \
  --patience 2 \
  --temperature 0.07 \
  --output-dir models/embeddings/lumafin-lacft-v1.0
echo "✓ Embedder fine-tuned"
echo

# Step 3: Seed database with labeled examples
echo "[3/6] Seeding database and building FAISS index..."
PYTHONPATH=. python scripts/seed_database.py --csv data/merged_training.csv
echo "✓ Database seeded, FAISS index built"
echo

# Step 4: Train XGBoost reranker
echo "[4/6] Training XGBoost reranker..."
PYTHONPATH=. python scripts/train_reranker.py \
  --source csv \
  --csv data/merged_training.csv \
  --k 20 \
  --n-estimators 300 \
  --max-depth 5 \
  --learning-rate 0.05
echo "✓ Reranker trained"
echo

# Step 5: Evaluate retrieval baseline
echo "[5/6] Evaluating retrieval baseline..."
PYTHONPATH=. python scripts/evaluate.py \
  --source db \
  --limit 1000 \
  --mode retrieval \
  --output evaluation_results_retrieval.json
echo "✓ Retrieval evaluation complete"
echo

# Step 6: Evaluate full fusion pipeline
echo "[6/6] Evaluating full fusion pipeline..."
PYTHONPATH=. python scripts/evaluate.py \
  --source db \
  --limit 1000 \
  --mode fusion \
  --output evaluation_results_fusion.json
echo "✓ Fusion evaluation complete"
echo

echo "=== Training Pipeline Complete ==="
echo
echo "Results:"
echo "  - Retrieval: evaluation_results_retrieval.json"
echo "  - Fusion:    evaluation_results_fusion.json"
echo
echo "Next steps:"
echo "  - Review metrics in evaluation JSON files"
echo "  - Start API: PYTHONPATH=. uvicorn src.api.main:app --reload"
echo "  - Run demo:  streamlit run demo/streamlit_app.py"
