#!/bin/bash
# Comprehensive LumaFin Training Pipeline for >90% Accuracy

set -e

echo "🚀 Starting LumaFin Training Pipeline for >90% Accuracy"
echo "======================================================"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if services are running
print_status "Checking Docker services..."
if ! docker-compose ps | grep -q "healthy\|running"; then
    print_status "Starting Docker services..."
    docker-compose up -d
    sleep 15
fi

# Verify database connectivity
print_status "Verifying database connectivity..."
PYTHONPATH=. python -c "
from src.storage.database import SessionLocal
try:
    db = SessionLocal()
    result = db.execute('SELECT COUNT(*) FROM global_examples')
    count = result.fetchone()[0]
    print(f'✅ Database connected. Global examples: {count}')
    db.close()
except Exception as e:
    print(f'❌ Database connection failed: {e}')
    exit(1)
"

# Seed database if needed
print_status "Ensuring database is seeded..."
PYTHONPATH=. python -c "
from src.storage.database import SessionLocal
db = SessionLocal()
result = db.execute('SELECT COUNT(*) FROM global_examples')
count = result.fetchone()[0]
db.close()
if count < 100:
    print('Seeding database...')
    import subprocess
    subprocess.run(['PYTHONPATH=.', 'python', 'scripts/seed_database.py'], check=True)
else:
    print(f'Database already seeded with {count} examples')
"

# Train optimized reranker model
print_status "Training optimized XGBoost reranker..."
PYTHONPATH=. python -c "
import os
os.environ['TOKENIZERS_PARALLELISM'] = 'false'

from scripts.train_reranker import main
import sys

# Override sys.argv for training parameters
sys.argv = [
    'train_reranker.py',
    '--source', 'db',
    '--limit', '2000',
    '--k', '30',
    '--n-estimators', '1000',
    '--max-depth', '8',
    '--learning-rate', '0.03'
]

try:
    main()
    print('✅ Reranker training completed')
except Exception as e:
    print(f'❌ Reranker training failed: {e}')
    sys.exit(1)
"

# Run comprehensive evaluation
print_status "Running comprehensive evaluation..."
PYTHONPATH=. python scripts/evaluate.py --mode fusion --limit 1000 > evaluation_results_final.json

# Parse and display results
print_status "Final Evaluation Results:"
python3 -c "
import json
with open('evaluation_results_final.json', 'r') as f:
    results = json.load(f)

macro_f1 = results['macro']['f1'] * 100
micro_f1 = results['micro']['f1'] * 100

print(f'🎯 Macro F1 Score: {macro_f1:.2f}%')
print(f'📊 Micro F1 Score: {micro_f1:.2f}%')

if macro_f1 >= 90.0:
    print('🎉 SUCCESS: Achieved >90% macro F1 accuracy!')
elif macro_f1 >= 85.0:
    print('👍 GOOD: Achieved >85% macro F1 accuracy')
elif macro_f1 >= 80.0:
    print('🤔 DECENT: Achieved >80% macro F1 accuracy')
else:
    print('⚠️  NEEDS IMPROVEMENT: Macro F1 below 80%')

print('\n📈 Per-Class Performance:')
for category, metrics in results['per_class'].items():
    if metrics['support'] > 0:
        f1 = metrics['f1'] * 100
        support = metrics['support']
        print(f'  {category}: F1={f1:.1f}% (n={support})')
"

print_status "Training pipeline completed!"
print_status "Check evaluation_results_final.json for detailed metrics"