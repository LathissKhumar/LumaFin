#!/bin/bash
# LumaFin Setup Script
# Automates initial setup of LumaFin development environment

set -e  # Exit on error

echo "🚀 LumaFin Setup Script"
echo "======================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo "📋 Checking prerequisites..."
if ! command -v python3.11 &> /dev/null; then
    echo -e "${RED}❌ Python 3.11+ not found. Please install Python 3.11 or higher.${NC}"
    exit 1
fi
echo -e "${GREEN}✓${NC} Python 3.11+ found"

# Check Docker
if ! command -v docker &> /dev/null; then
    echo -e "${YELLOW}⚠ Docker not found. You'll need to install PostgreSQL and Redis manually.${NC}"
    USE_DOCKER=false
else
    echo -e "${GREEN}✓${NC} Docker found"
    USE_DOCKER=true
fi

# Check Docker Compose
if ! command -v docker-compose &> /dev/null; then
    if [ "$USE_DOCKER" = true ]; then
        echo -e "${YELLOW}⚠ docker-compose not found. Trying 'docker compose' instead...${NC}"
        DOCKER_COMPOSE="docker compose"
    fi
else
    DOCKER_COMPOSE="docker-compose"
    echo -e "${GREEN}✓${NC} docker-compose found"
fi

echo ""

# Step 1: Create virtual environment
echo "1️⃣  Creating virtual environment..."
if [ -d ".venv" ]; then
    echo -e "${YELLOW}⚠ Virtual environment already exists. Skipping...${NC}"
else
    python3.11 -m venv .venv
    echo -e "${GREEN}✓${NC} Virtual environment created"
fi

# Step 2: Activate and install dependencies
echo ""
echo "2️⃣  Installing Python dependencies..."
source .venv/bin/activate
pip install --upgrade pip > /dev/null 2>&1
pip install -e . > /dev/null 2>&1
echo -e "${GREEN}✓${NC} Dependencies installed"

# Step 3: Create .env file
echo ""
echo "3️⃣  Creating .env file..."
if [ -f ".env" ]; then
    echo -e "${YELLOW}⚠ .env already exists. Skipping...${NC}"
else
    cat > .env << EOF
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/lumafin
REDIS_URL=redis://localhost:6379/0

# Models
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL_PATH=models/reranker/xgb_reranker.json
FAISS_INDEX_PATH=models/faiss_index.bin
FAISS_METADATA_PATH=models/faiss_metadata.pkl

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=TEXT

# API (optional - uncomment to enable)
# SECRET_KEY=$(openssl rand -hex 32)
# BEARER_TOKEN=$(openssl rand -hex 16)
# ENABLE_RATE_LIMIT=false

# Celery (optional)
CELERY_DISABLE_BEAT=false
FEEDBACK_INTERVAL_MINUTES=5
EOF
    echo -e "${GREEN}✓${NC} .env file created"
fi

# Step 4: Start Docker services
if [ "$USE_DOCKER" = true ]; then
    echo ""
    echo "4️⃣  Starting Docker services (PostgreSQL + Redis)..."
    $DOCKER_COMPOSE up -d
    echo -e "${GREEN}✓${NC} Docker services started"
    
    # Wait for services to be ready
    echo "   Waiting for services to be ready..."
    sleep 5
else
    echo ""
    echo "4️⃣  ${YELLOW}Skipping Docker services (Docker not available)${NC}"
    echo "   Please ensure PostgreSQL and Redis are running manually."
fi

# Step 5: Create database schema
echo ""
echo "5️⃣  Creating database schema..."
if PYTHONPATH=. python3 -c "from src.storage.database import Base, engine; Base.metadata.create_all(bind=engine)" 2>/dev/null; then
    echo -e "${GREEN}✓${NC} Database schema created"
else
    echo -e "${YELLOW}⚠ Could not create schema (database may not be ready). You can run this manually:${NC}"
    echo "   PYTHONPATH=. python -c \"from src.storage.database import Base, engine; Base.metadata.create_all(bind=engine)\""
fi

# Step 6: Create directories
echo ""
echo "6️⃣  Creating necessary directories..."
mkdir -p data models/embeddings models/reranker
echo -e "${GREEN}✓${NC} Directories created"

# Step 7: Generate sample data
echo ""
echo "7️⃣  Generating sample transaction data..."
if [ -f "data/global_examples.csv" ]; then
    echo -e "${YELLOW}⚠ Sample data already exists. Skipping...${NC}"
else
    if PYTHONPATH=. python scripts/generate_sample_data.py --count 500 --output data/global_examples.csv > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Sample data generated (500 transactions)"
    else
        echo -e "${YELLOW}⚠ Could not generate sample data. You can run this manually:${NC}"
        echo "   PYTHONPATH=. python scripts/generate_sample_data.py --count 500"
    fi
fi

# Step 8: Seed database
echo ""
echo "8️⃣  Seeding database and building FAISS index..."
if [ -f "models/faiss_index.bin" ]; then
    echo -e "${YELLOW}⚠ FAISS index already exists. Skipping...${NC}"
else
    if PYTHONPATH=. python scripts/seed_database.py > /dev/null 2>&1; then
        echo -e "${GREEN}✓${NC} Database seeded and index built"
    else
        echo -e "${YELLOW}⚠ Could not seed database. You can run this manually:${NC}"
        echo "   PYTHONPATH=. python scripts/seed_database.py"
    fi
fi

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo "=========================================="
echo ""
echo "📚 Next steps:"
echo ""
echo "1. Start the API server:"
echo "   source .venv/bin/activate"
echo "   PYTHONPATH=. uvicorn src.api.main:app --reload"
echo ""
echo "2. Test categorization:"
echo "   curl -X POST http://localhost:8000/categorize \\"
echo "     -H \"Content-Type: application/json\" \\"
echo "     -d '{\"merchant\": \"Starbucks\", \"amount\": 5.50}'"
echo ""
echo "3. (Optional) Train models for better accuracy:"
echo "   PYTHONPATH=. python scripts/train_reranker.py --limit 500"
echo ""
echo "4. (Optional) Launch demo UI:"
echo "   PYTHONPATH=. streamlit run demo/streamlit_app.py"
echo ""
echo "5. (Optional) Enable feedback loop:"
echo "   celery -A src.training.celery_app worker --loglevel=info"
echo ""
echo "📖 Read QUICKSTART.md for more details!"
echo ""
