# LumaFin Hybrid Adaptive System (LHAS) – Production AI Transaction Categorization

Build a production-grade transaction categorization engine that fuses **global ML classification** (Label-Aware Contrastive Fine-Tuning, FAISS retrieval, reranker) with **AMPT's user-specific clustering engine** (personalized micro-categories), **rule-based overrides**, and **comprehensive explainability**. Start as modular monolith with microservice boundaries, implement hierarchical decision pipeline (Rules → Personal Centroids → Retrieval+Rerank → Fallback), enable continuous learning via feedback loop, and maintain privacy-first architecture with federated learning readiness.

## Implementation Steps

### 1. Initialize project structure and core dependencies

Create `/src` with modules: `preprocessing/`, `embedder/`, `indexer/`, `retrieval/`, `reranker/`, `clustering/` (AMPT), `rules/`, `explainer/`, `fusion/`, `api/`; add `/data` (global_examples.csv, taxonomy.json), `/models`, `/demo` (Streamlit), `/tests`, `/k8s`; initialize `pyproject.toml` with `sentence-transformers==2.2.2`, `faiss-cpu==1.7.4`, `hdbscan==0.8.33`, `torch==2.1.0`, `transformers==4.35.0`, `fastapi==0.104.1`, `shap==0.43.0`, `scikit-learn==1.3.2`, `psycopg2-binary==2.9.9`, `pgvector==0.2.3`, `celery==5.3.4`, `redis==5.0.1`, `pydantic==2.5.0`; create PostgreSQL schema in `src/storage/schema.sql` with tables `users`, `transactions` (with `vector(384)` for embeddings), `personal_centroids`, `global_taxonomy`, `global_examples`, `rules`, `feedback_queue`; setup Docker Compose for local PostgreSQL+Redis+pgvector; add `.env.example` with `DATABASE_URL`, `REDIS_URL`, `MODEL_PATH`, `SECRET_KEY`.

### 2. Build preprocessing and embedding pipeline with L-A CFT

Implement merchant normalization in `src/preprocessing/normalize.py` (lowercase, punctuation strip, abbreviation dict: {"SQ *":"square", "AMZN":"amazon"}, remove location codes, MCC lookup); create Pydantic models in `src/models.py` for `Transaction`, `Category`, `PersonalCentroid`, `Prediction`, `Explanation`; wrap `sentence-transformers/all-MiniLM-L6-v2` in `src/embedder/encoder.py` with batch encoding and single-transaction methods; implement Label-Aware Contrastive Fine-Tuning trainer in `src/embedder/train_lacft.py` using PyTorch with contrastive loss (anchor=transaction, positive=same category, negative=different category), train on seed dataset with 1000 labeled examples, save fine-tuned model to `/models/embeddings/lumafin-lacft-v1.0/`; add amount/time bucketing utilities (amount: [0-10, 10-50, 50-100, 100-500, 500+], time: hour bins, weekday/weekend).

### 3. Implement FAISS global retrieval and rule engine

Build FAISS Flat index manager in `src/indexer/faiss_builder.py` that loads `global_examples` table, encodes with fine-tuned embeddings, creates `IndexFlatIP` (inner product for cosine similarity with normalized vectors), provides `add_vectors()`, `search(query_vector, k=20)`, `save()/load()` methods; create retrieval service in `src/retrieval/service.py` that takes transaction embedding, queries FAISS for top-K global examples with scores, returns candidates with category labels; implement rule engine in `src/rules/engine.py` loading patterns from `rules` table (regex matching: "(?i)netflix" → Entertainment, "(?i)atm" → Cash), apply priority-ordered rule matching, return immediate category if deterministic rule fires with confidence=1.0; create YAML config `taxonomy/rules.yaml` for non-DB rule definitions.

### 4. Build AMPT clustering engine and personal centroid system

Implement HDBSCAN clustering service in `src/clustering/ampt_engine.py` that fetches user transaction history (minimum 50 transactions), extracts embeddings, runs `HDBSCAN(min_cluster_size=5, min_samples=3, metric='cosine')`, evaluates cluster quality (silhouette score > 0.4, semantic distance from global categories > 0.3 threshold), filters noise clusters; create category name generator in `src/clustering/name_generator.py` extracting dominant keywords (TF-IDF on cluster transactions), merchant patterns (frequency analysis), temporal signals (time-of-day mode, weekday concentration), amount statistics (mean, std), generate labels like "Daily Morning Coffee" or "Weekend Groceries"; compute centroid embeddings (mean of cluster vectors), store in `personal_centroids` table with metadata JSON `{merchant_pattern, time_pattern, amount_range, creation_date, num_transactions}`; implement centroid matcher in `src/clustering/centroid_matcher.py` that computes cosine similarity between query embedding and user's centroids, returns personal category if similarity > 0.80 AND centroid has ≥5 supporting transactions.

### 5. Develop reranker and prediction fusion layer

Build cross-encoder reranker in `src/reranker/model.py` using `cross-encoder/ms-marco-MiniLM-L-6-v2`, engineer features: FAISS similarity scores (max, mean, min of top-K), rule match boosts (+0.3 for merchant exact match, +0.2 for MCC alignment, +0.1 for amount proximity), centroid distance (if personal category exists, add as feature), numeric buckets (log amount, hour bin, day-of-week one-hot), category vote distribution from top-K; train lightweight XGBoost model on engineered features + cross-encoder scores, output category probabilities with Platt scaling for calibrated confidence; implement fusion engine in `src/fusion/decision.py` with hierarchical logic: **(1)** check rule engine first (if match, return with conf=1.0), **(2)** check personal centroids (if strong match >0.80, override with conf=0.85-0.95), **(3)** run FAISS retrieval + reranker (return global category with calibrated confidence), **(4)** fallback to "Uncategorized" if conf < 0.50; add confidence thresholds config (high: >0.85 auto-accept, medium: 0.50-0.85 show to user, low: <0.50 require manual).

### 6. Create explainability module and FastAPI service

Build explainer in `src/explainer/explain.py` generating multi-source explanations: **(A)** nearest examples (top-3 from FAISS with merchant, amount, date, similarity score), **(B)** SHAP feature attributions using `shap.TreeExplainer` on reranker XGBoost model (show importance of merchant similarity, amount bucket, time features), **(C)** rule trace (which rules evaluated, which fired), **(D)** cluster coherence (if personal category, show silhouette score + how many similar transactions in cluster), **(E)** natural language summary template ("Categorized as {category} because it matches your pattern of {pattern} with {confidence}% confidence"); create FastAPI app in `src/api/main.py` with endpoints: `POST /categorize` (input: transaction text/amount/date/user_id, output: category/confidence/explanation/is_personal), `POST /feedback` (submit correction, update centroid), `GET /taxonomy/personal/{user_id}` (list user micro-categories), `POST /taxonomy/retrain/{user_id}` (trigger AMPT re-clustering), `GET /taxonomy/global` (browse hierarchy); add JWT token auth, rate limiting (100 req/min), input validation, TLS config, CORS for React frontend.

### 7. Implement feedback loop and incremental training pipeline

Create Celery worker in `src/training/feedback_worker.py` that polls `feedback_queue` table, processes user corrections by: **(A)** updating personal centroid (moving average: `new_centroid = 0.9 * old_centroid + 0.1 * correction_embedding`), **(B)** adding corrected transaction to global examples if correction differs from global prediction (potential new training data), **(C)** flagging user for AMPT re-clustering if >20 new corrections since last run; build scheduled trainer in `src/training/incremental.py` running nightly: refresh FAISS index with new global examples, re-fit reranker XGBoost using accumulated feedback (warm start), optionally perform one epoch of L-A CFT fine-tuning on high-confidence corrections, update model version in database, log metrics (macro F1, per-category precision/recall) to MLflow or CSV; implement active learning query strategy in `src/training/active_learning.py` selecting transactions with: uncertainty (confidence 0.45-0.65), representativeness (diverse embeddings via k-means centers), disagreement (rules vs. retrieval vs. centroids differ), surface top-10 to user for labeling.

### 8. Add evaluation framework and Streamlit demo

Create evaluation script `scripts/evaluate.py` with stratified train/val/test split (70/15/15), compute macro F1, per-class precision/recall, confusion matrix using `sklearn.metrics`, fairness checks (group by amount bucket [<$10, $10-100, >$100] and compute F1 variance, check demographic parity), calibration plot (predicted confidence vs. actual accuracy), log to `evaluation_results.json`; implement per-user adaptation metrics (accuracy over first N transactions, personal category coverage rate); build Streamlit UI in `demo/streamlit_app.py` with: CSV upload widget, transaction table display with predicted categories, correction interface (dropdown to fix category), personal taxonomy visualization (treemap of user categories), confidence distribution histogram, explanation panel showing SHAP values + nearest examples, feedback submission button; add Docker Compose orchestration in `docker-compose.yml` for PostgreSQL, Redis, FastAPI service, Celery worker, Streamlit frontend.

## Further Considerations

### 1. Microservices decomposition timeline

Current modular monolith has clear service boundaries (preprocessing, embedder, retrieval, reranker, AMPT, rules, explainer) making future extraction straightforward; consider decomposing after reaching 1000+ concurrent users or when AMPT clustering becomes CPU bottleneck; use gRPC for inter-service communication (faster than REST), deploy each service in separate Kubernetes pods with horizontal autoscaling, implement circuit breakers with Istio/Envoy for fault tolerance, add distributed tracing with Jaeger. Should we containerize each module now with individual Dockerfiles to enable gradual rollout, or keep monolithic deployment until performance metrics justify splitting?

### 2. Federated learning implementation approach

For privacy-preserving multi-user learning, implement federated averaging where each user's AMPT engine computes local centroid updates, encrypts gradient deltas using homomorphic encryption (PySyft or TenSEAL), sends to central aggregator that updates global taxonomy without seeing raw transactions; requires: secure aggregation protocol (minimum 10 users to prevent inference attacks), differential privacy noise addition (epsilon=1.0 for strong privacy), local model caching to minimize network traffic. Should we build federated prototype in Phase 4 as proof-of-concept, or defer until actual multi-tenant production deployment demand arises?

### 3. Explainability depth and compliance

Current design provides SHAP attributions for reranker, nearest examples for retrieval, rule traces, and cluster coherence metrics; for financial regulations (GDPR Article 22 "right to explanation"), consider adding: counterfactual explanations ("if merchant was X instead of Y, category would be Z"), feature importance visualization (horizontal bar charts in UI), audit trail storage (every prediction + explanation logged for 7 years), model card documentation (training data, performance by demographic, known limitations). Should explanations be generated synchronously (adds 100-200ms latency) or asynchronously via background job with explanation caching for common patterns?

---

## Architecture Overview

### Unified System Design

```
┌─────────────────────────────────────────────────────────────────┐
│                         API GATEWAY                              │
│              (FastAPI, Auth, Rate Limiting)                      │
└────────────┬────────────────────────────────────────────────────┘
             │
   ┌─────────┴──────────┐
   │                     │
┌──▼─────────────┐  ┌───▼────────────────┐
│  USER SERVICE  │  │ TRANSACTION INPUT  │
│  (Profile/Auth)│  │  (Preprocessing)   │
└────────────────┘  └───┬────────────────┘
                        │
                        │ Cleaned Transaction
                        │
                ┌───────▼────────────┐
                │  EMBEDDING SERVICE │ ← Fine-tuned L-A CFT Model
                │ (sentence-trans.)  │
                └───┬────────────────┘
                    │
                    │ 384-dim vector
                    │
        ┌───────────┴────────────┐
        │                        │
┌───────▼─────────┐      ┌──────▼─────────────┐
│  RULE ENGINE    │      │  AMPT CLUSTERING   │
│  (Deterministic)│      │  SERVICE           │
└───┬─────────────┘      └──────┬─────────────┘
    │                            │
    │ If matched, DONE           │ Check personal centroids
    │                            │
    │                    ┌───────▼────────────┐
    │                    │  RETRIEVAL SERVICE │
    │                    │  (FAISS Global)    │
    │                    └───────┬────────────┘
    │                            │
    │                            │ Top-k candidates
    │                            │
    │                    ┌───────▼────────────┐
    │                    │  RERANKER SERVICE  │
    │                    │  (Cross-encoder +  │
    │                    │   centroid boost)  │
    │                    └───────┬────────────┘
    │                            │
    └────────────┬───────────────┘
                 │
         ┌───────▼─────────────┐
         │ PREDICTION FUSION   │ ← Combines rules, centroids, retrieval
         └───────┬─────────────┘
                 │
         ┌───────▼─────────────┐
         │ EXPLAINABILITY SVC  │ ← SHAP + Examples + Cluster Coherence
         └───────┬─────────────┘
                 │
         ┌───────▼─────────────┐
         │   RESPONSE WITH:    │
         │ - Category          │
         │ - Confidence        │
         │ - Explanations      │
         │ - Similar Examples  │
         └───────┬─────────────┘
                 │
         ┌───────▼─────────────┐
         │  FEEDBACK LOOP      │
         │  (Active Learning)  │
         │                     │
         │ ┌─────────────────┐ │
         │ │ Update Centroids│ │ ← User corrections
         │ │ Retrain L-A CFT │ │ ← Periodic fine-tuning
         │ │ Refresh FAISS   │ │ ← Index new examples
         │ │ Expand KG       │ │ ← New categories
         │ └─────────────────┘ │
         └─────────────────────┘
```

### Hierarchical Decision Pipeline

**Priority Order:**
1. **Rules First**: Deterministic patterns (e.g., "NETFLIX" → Entertainment) - Confidence: 1.0
2. **Personal Centroids Second**: User-specific micro-categories (similarity > 0.80) - Confidence: 0.85-0.95
3. **Retrieval + Rerank Third**: Global knowledge base via FAISS - Confidence: varies
4. **Fallback Fourth**: "Uncategorized" if confidence < 0.50

### Database Schema (PostgreSQL + pgvector)

```sql
-- Users table
CREATE TABLE users (
    user_id SERIAL PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    settings JSONB DEFAULT '{}'
);

-- Transactions table with vector embeddings
CREATE TABLE transactions (
    transaction_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(user_id),
    raw_text TEXT NOT NULL,
    cleaned_text TEXT,
    amount DECIMAL(10, 2),
    date TIMESTAMP,
    embedding vector(384),
    predicted_category VARCHAR(100),
    confidence FLOAT,
    is_corrected BOOLEAN DEFAULT FALSE,
    correct_category VARCHAR(100),
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_user_transactions ON transactions(user_id, date DESC);
CREATE INDEX idx_embedding ON transactions USING ivfflat (embedding vector_cosine_ops);

-- Personal centroids for AMPT
CREATE TABLE personal_centroids (
    centroid_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(user_id),
    category_label VARCHAR(100) NOT NULL,
    centroid_vector vector(384),
    num_transactions INT DEFAULT 1,
    metadata JSONB,
    last_updated TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id, category_label)
);

CREATE INDEX idx_user_centroids ON personal_centroids(user_id);
CREATE INDEX idx_centroid_vector ON personal_centroids USING ivfflat (centroid_vector vector_cosine_ops);

-- Global taxonomy
CREATE TABLE global_taxonomy (
    category_id SERIAL PRIMARY KEY,
    category_name VARCHAR(100) UNIQUE NOT NULL,
    parent_category_id INT REFERENCES global_taxonomy(category_id),
    level INT,
    description TEXT,
    example_keywords TEXT[]
);

-- Global labeled examples
CREATE TABLE global_examples (
    example_id SERIAL PRIMARY KEY,
    category_id INT REFERENCES global_taxonomy(category_id),
    text TEXT NOT NULL,
    embedding vector(384),
    source VARCHAR(50) DEFAULT 'seed',
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_category_examples ON global_examples(category_id);
CREATE INDEX idx_example_embedding ON global_examples USING ivfflat (embedding vector_cosine_ops);

-- Rule engine patterns
CREATE TABLE rules (
    rule_id SERIAL PRIMARY KEY,
    pattern TEXT NOT NULL,
    category_name VARCHAR(100) NOT NULL,
    priority INT DEFAULT 1,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Feedback queue for continuous learning
CREATE TABLE feedback_queue (
    feedback_id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(user_id),
    transaction_id INT REFERENCES transactions(transaction_id),
    predicted_category VARCHAR(100),
    correct_category VARCHAR(100),
    processed BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX idx_unprocessed_feedback ON feedback_queue(processed, created_at);
```

## Technology Stack

| Layer | Technology | Justification |
|-------|-----------|---------------|
| **API** | FastAPI | Async, auto-docs, type hints |
| **Database** | PostgreSQL + pgvector | Mature, vector support, ACID |
| **Vector Search** | FAISS | Fastest in-memory, battle-tested |
| **Embeddings** | sentence-transformers (all-MiniLM-L6-v2) | 384-dim, 120M params, fast |
| **Clustering** | HDBSCAN | Density-based, auto-determines K |
| **Reranker** | Cross-encoder (ms-marco-MiniLM-L-6-v2) | Better than bi-encoder alone |
| **Rules** | YAML + regex | Flexible, non-programmer editable |
| **Explainability** | SHAP + Custom | Model-agnostic |
| **Training** | PyTorch + sentence-transformers | Fine-tuning L-A CFT |
| **Queue** | Celery + Redis | Background feedback processing |
| **Frontend** | Streamlit (demo) → React (prod) | Rapid prototyping → Scale |
| **Deployment** | Docker + Kubernetes | Microservices orchestration |
| **Monitoring** | Prometheus + Grafana | Metrics, alerts |

## Key Innovations

1. **Hybrid Taxonomy**: Global categories + user-discovered micro-categories coexist
2. **Hierarchical Decision Making**: Rules > Centroids > Retrieval > Fallback
3. **Dual Explainability**: SHAP for models, examples for clusters
4. **Zero-Effort Personalization**: AMPT clustering with no manual labeling
5. **Privacy-First**: User-level isolation + federated learning ready
6. **Label-Aware Contrastive Fine-Tuning**: Semantic embeddings optimized for financial transactions
7. **Confidence Calibration**: Platt scaling ensures reliable uncertainty estimates
8. **Active Learning Integration**: Prioritizes high-impact examples for human labeling

## Implementation Phases

### MVP (Weeks 1-4): Core Categorization
- Preprocessing, Embedding, FAISS Retrieval, Basic Reranker
- FastAPI endpoints, Streamlit UI
- **Goal**: 80%+ accuracy on standard categories

### Phase 2 (Weeks 5-8): Personalization (AMPT Core)
- HDBSCAN clustering, Personal centroids, Centroid override logic
- Feedback loop, Confidence calibration
- **Goal**: User-specific categories with 90%+ accuracy

### Phase 3 (Weeks 9-12): Advanced Features
- Rule engine, Label-Aware CFT, SHAP explainability
- Upgraded reranker, Knowledge graph
- **Goal**: Production-grade accuracy and explainability

### Phase 4 (Weeks 13-16): Scale & Privacy
- Microservices refactor, User isolation
- Federated learning, Active learning
- **Goal**: Multi-user, production-ready system

## Project Structure

```
/src
  /preprocessing        # Transaction normalization
  /embedder            # Sentence-transformers wrapper + L-A CFT trainer
  /indexer             # FAISS index builder
  /retrieval           # Query FAISS, return top-k
  /reranker            # Cross-encoder + feature engineering
  /clustering          # HDBSCAN AMPT engine
  /rules               # Pattern matching engine
  /explainer           # SHAP + examples + cluster metrics
  /fusion              # Hierarchical decision combiner
  /api                 # FastAPI endpoints
  /training            # Feedback worker, incremental trainer, active learning
  /storage             # Database models, migrations
  models.py            # Pydantic schemas

/data
  global_examples.csv  # Seed labeled transactions
  taxonomy.json        # Category hierarchy

/models
  /embeddings          # Fine-tuned sentence-transformers
  /reranker            # XGBoost checkpoints

/demo
  streamlit_app.py     # Interactive UI

/tests
  /unit
  /integration

/k8s
  /deployments         # Kubernetes manifests

/scripts
  evaluate.py          # Model evaluation
  seed_database.py     # Initialize with sample data

/taxonomy
  rules.yaml           # Deterministic rule patterns

docker-compose.yml     # Local dev environment
pyproject.toml         # Dependencies
.env.example           # Configuration template
README.md              # Project documentation
```

## Success Metrics

### MVP Success:
- ✅ Categorizes 10 transactions in <2 seconds
- ✅ 80%+ accuracy on standard categories
- ✅ Shows 3 similar transactions as explanations
- ✅ User corrections update system

### Phase 2 (AMPT):
- ✅ Personal categories emerge after 50 transactions
- ✅ 90%+ accuracy within user context
- ✅ Centroid updates in <1 second

### Phase 3 (Advanced):
- ✅ Rules override 100% correctly
- ✅ SHAP explanations clear to non-technical users
- ✅ Fine-tuned embeddings improve accuracy by 5%+

### Phase 4 (Scale):
- ✅ Supports 10,000 concurrent users
- ✅ 99.9% uptime
- ✅ Federated learning without data leakage
