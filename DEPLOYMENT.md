# LumaFin Deployment Guide

## Overview

This guide covers deploying LumaFin to production using Docker Compose and Kubernetes.

## Prerequisites

- Docker & Docker Compose
- Kubernetes cluster (for K8s deployment)
- PostgreSQL 14+ with pgvector extension
- Redis 7+
- 2+ GB RAM, 2+ CPU cores recommended

## Docker Compose Deployment (Recommended for Small-Scale)

### 1. Production docker-compose.yml

```yaml
version: '3.8'

services:
  postgres:
    image: ankane/pgvector:latest
    environment:
      POSTGRES_DB: lumafin
      POSTGRES_USER: lumafin
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./src/storage/schema.sql:/docker-entrypoint-initdb.d/schema.sql
    ports:
      - "5432:5432"
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U lumafin"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --appendonly yes
    volumes:
      - redis_data:/data
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 3s
      retries: 5
    restart: unless-stopped

  api:
    build:
      context: .
      dockerfile: Dockerfile.api
    environment:
      DATABASE_URL: postgresql://lumafin:${POSTGRES_PASSWORD}@postgres:5432/lumafin
      REDIS_URL: redis://redis:6379/0
      SECRET_KEY: ${SECRET_KEY}
      BEARER_TOKEN: ${BEARER_TOKEN}
      LOG_FORMAT: JSON
      LOG_LEVEL: INFO
    ports:
      - "8000:8000"
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    command: uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --workers 2

  celery_worker:
    build:
      context: .
      dockerfile: Dockerfile.api
    environment:
      DATABASE_URL: postgresql://lumafin:${POSTGRES_PASSWORD}@postgres:5432/lumafin
      REDIS_URL: redis://redis:6379/0
      LOG_FORMAT: JSON
      LOG_LEVEL: INFO
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    command: celery -A src.training.celery_app worker --loglevel=info --concurrency=2

  celery_beat:
    build:
      context: .
      dockerfile: Dockerfile.api
    environment:
      DATABASE_URL: postgresql://lumafin:${POSTGRES_PASSWORD}@postgres:5432/lumafin
      REDIS_URL: redis://redis:6379/0
      LOG_FORMAT: JSON
      LOG_LEVEL: INFO
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    restart: unless-stopped
    command: celery -A src.training.celery_app beat --loglevel=info

volumes:
  postgres_data:
  redis_data:
```

### 2. Create Dockerfile.api

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY pyproject.toml .
COPY requirements.txt* .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e .

# Copy application code
COPY src/ ./src/
COPY models/ ./models/
COPY scripts/ ./scripts/

# Create non-root user
RUN useradd -m -u 1000 lumafin && chown -R lumafin:lumafin /app
USER lumafin

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### 3. Environment Setup

Create `.env.prod`:

```bash
# Database
POSTGRES_PASSWORD=<strong-random-password>
DATABASE_URL=postgresql://lumafin:${POSTGRES_PASSWORD}@postgres:5432/lumafin

# Redis
REDIS_URL=redis://redis:6379/0

# Security
SECRET_KEY=<generate-with-openssl-rand-hex-32>
BEARER_TOKEN=<generate-strong-token>

# Models
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANKER_MODEL_PATH=models/reranker/xgb_reranker.json
FAISS_INDEX_PATH=models/faiss_index.bin

# API
ENABLE_RATE_LIMIT=true
LOG_FORMAT=JSON
LOG_LEVEL=INFO

# Celery
FEEDBACK_INTERVAL_MINUTES=5
```

### 4. Deploy

```bash
# Build images
docker-compose -f docker-compose.yml build

# Start services
docker-compose -f docker-compose.yml --env-file .env.prod up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f api

# Initialize database
docker-compose exec api python scripts/seed_database.py
```

## Kubernetes Deployment (Scalable Production)

### 1. Create ConfigMap

`k8s/configmap.yml`:

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: lumafin-config
  namespace: lumafin
data:
  LOG_FORMAT: "JSON"
  LOG_LEVEL: "INFO"
  FEEDBACK_INTERVAL_MINUTES: "5"
  EMBEDDING_MODEL: "sentence-transformers/all-MiniLM-L6-v2"
```

### 2. Create Secrets

```bash
kubectl create secret generic lumafin-secrets \
  --from-literal=database-url='postgresql://user:pass@postgres:5432/lumafin' \
  --from-literal=redis-url='redis://redis:6379/0' \
  --from-literal=secret-key='<strong-key>' \
  --from-literal=bearer-token='<strong-token>' \
  --namespace=lumafin
```

### 3. Deploy PostgreSQL

`k8s/postgres.yml`:

```yaml
apiVersion: apps/v1
kind: StatefulSet
metadata:
  name: postgres
  namespace: lumafin
spec:
  serviceName: postgres
  replicas: 1
  selector:
    matchLabels:
      app: postgres
  template:
    metadata:
      labels:
        app: postgres
    spec:
      containers:
      - name: postgres
        image: ankane/pgvector:latest
        ports:
        - containerPort: 5432
        env:
        - name: POSTGRES_DB
          value: lumafin
        - name: POSTGRES_PASSWORD
          valueFrom:
            secretKeyRef:
              name: lumafin-secrets
              key: postgres-password
        volumeMounts:
        - name: postgres-storage
          mountPath: /var/lib/postgresql/data
  volumeClaimTemplates:
  - metadata:
      name: postgres-storage
    spec:
      accessModes: ["ReadWriteOnce"]
      resources:
        requests:
          storage: 20Gi
---
apiVersion: v1
kind: Service
metadata:
  name: postgres
  namespace: lumafin
spec:
  selector:
    app: postgres
  ports:
  - port: 5432
    targetPort: 5432
  clusterIP: None
```

### 4. Deploy API

`k8s/api.yml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lumafin-api
  namespace: lumafin
spec:
  replicas: 3
  selector:
    matchLabels:
      app: lumafin-api
  template:
    metadata:
      labels:
        app: lumafin-api
    spec:
      containers:
      - name: api
        image: your-registry/lumafin-api:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: lumafin-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: lumafin-secrets
              key: redis-url
        envFrom:
        - configMapRef:
            name: lumafin-config
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "1Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
---
apiVersion: v1
kind: Service
metadata:
  name: lumafin-api
  namespace: lumafin
spec:
  selector:
    app: lumafin-api
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

### 5. Deploy Celery Workers

`k8s/celery.yml`:

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: lumafin-worker
  namespace: lumafin
spec:
  replicas: 2
  selector:
    matchLabels:
      app: lumafin-worker
  template:
    metadata:
      labels:
        app: lumafin-worker
    spec:
      containers:
      - name: worker
        image: your-registry/lumafin-api:latest
        command: ["celery", "-A", "src.training.celery_app", "worker", "--loglevel=info", "--concurrency=2"]
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: lumafin-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: lumafin-secrets
              key: redis-url
        envFrom:
        - configMapRef:
            name: lumafin-config
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
```

### 6. Deploy

```bash
# Create namespace
kubectl create namespace lumafin

# Apply configurations
kubectl apply -f k8s/configmap.yml
kubectl apply -f k8s/postgres.yml
kubectl apply -f k8s/api.yml
kubectl apply -f k8s/celery.yml

# Check status
kubectl get pods -n lumafin
kubectl get svc -n lumafin

# View logs
kubectl logs -f deployment/lumafin-api -n lumafin
```

## Monitoring & Observability

### Prometheus Metrics

Add to `src/api/main.py`:

```python
from prometheus_client import Counter, Histogram, make_asgi_app

# Metrics
categorization_requests = Counter('categorization_requests_total', 'Total categorization requests')
categorization_latency = Histogram('categorization_latency_seconds', 'Categorization latency')

# Mount at /metrics
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

### Health Checks

```python
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.get("/ready")
async def readiness_check():
    # Check database, Redis, models loaded
    try:
        db = SessionLocal()
        db.execute(text("SELECT 1"))
        db.close()
        return {"status": "ready"}
    except Exception as e:
        raise HTTPException(status_code=503, detail="Not ready")
```

## Performance Tuning

### Database Connection Pooling

Update `src/storage/database.py`:

```python
engine = create_engine(
    DATABASE_URL,
    pool_size=20,
    max_overflow=40,
    pool_pre_ping=True,
    pool_recycle=3600
)
```

### API Workers

```bash
# Production: 2-4 workers per CPU core
uvicorn src.api.main:app --workers 4 --host 0.0.0.0 --port 8000
```

### Caching

Add Redis caching for frequent queries:

```python
from redis import Redis

cache = Redis.from_url(REDIS_URL)

def get_category_cached(merchant: str, amount: float):
    key = f"cat:{merchant}:{amount}"
    cached = cache.get(key)
    if cached:
        return json.loads(cached)
    
    result = categorize(merchant, amount)
    cache.setex(key, 3600, json.dumps(result))  # 1 hour TTL
    return result
```

## Backup & Recovery

### Database Backup

```bash
# Automated backups (cron)
docker-compose exec postgres pg_dump -U lumafin lumafin > backup_$(date +%Y%m%d).sql

# Restore
docker-compose exec -T postgres psql -U lumafin lumafin < backup_20250115.sql
```

### Model Backup

```bash
# Backup models directory
tar -czf models_backup_$(date +%Y%m%d).tar.gz models/

# Upload to S3
aws s3 cp models_backup_20250115.tar.gz s3://your-bucket/lumafin/models/
```

## Security Hardening

### 1. Enable TLS

Use nginx reverse proxy with Let's Encrypt:

```nginx
server {
    listen 443 ssl http2;
    server_name api.lumafin.com;
    
    ssl_certificate /etc/letsencrypt/live/api.lumafin.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/api.lumafin.com/privkey.pem;
    
    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

### 2. Rate Limiting

Already implemented in API. Configure via env:

```bash
ENABLE_RATE_LIMIT=true
RATE_LIMIT_PER_MINUTE=100
```

### 3. Network Isolation

Use Docker networks or K8s network policies to isolate services.

## Scaling Considerations

- **API**: Horizontal scaling via load balancer (3-10+ replicas)
- **Celery Workers**: Scale based on queue length (2-5 workers)
- **PostgreSQL**: Consider read replicas for high read load
- **Redis**: Cluster mode for high-volume feedback processing
- **FAISS Index**: Load into memory on each API pod (384MB for 100K vectors)

## Cost Optimization

- Use spot instances for Celery workers
- Schedule model training during off-peak hours
- Implement caching to reduce database queries
- Use serverless functions for infrequent tasks

## Troubleshooting

### High Memory Usage

```bash
# Check memory per service
docker stats

# Limit Celery worker memory
celery -A src.training.celery_app worker --max-memory-per-child=500000
```

### Slow Categorization

- Check FAISS index loaded correctly
- Enable connection pooling
- Add result caching
- Scale API horizontally

### Database Connection Exhaustion

- Increase `pool_size` in SQLAlchemy
- Add connection pooling middleware (PgBouncer)
- Close sessions properly in endpoints

---

For support, open an issue on [GitHub](https://github.com/LathissKhumar/LumaFin/issues).
