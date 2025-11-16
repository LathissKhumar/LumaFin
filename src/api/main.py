from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import text
import os
from dotenv import load_dotenv

from src.models import Transaction, Prediction, Category, Explanation
from src.preprocessing.normalize import get_time_features, normalize_label
from src.storage.database import get_db
from src.embedder.encoder import TransactionEmbedder
from src.rules.engine import rule_engine
from src.retrieval.service import get_retrieval_service
from src.fusion.decision import decide
from src.explainer.explain import build_explanation
from src.clustering.ampt_engine import AMPTClusteringEngine
from src.utils.logger import get_logger
from sqlalchemy import text
from src.clustering.centroid_matcher import match_personal_category
from src.utils.logger import get_logger

load_dotenv()

app = FastAPI(
    title="LumaFin Transaction Categorization API",
    description="Hybrid adaptive system for financial transaction categorization",
    version="0.1.0"
)

# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize embedder and retrieval (loads once at startup)
embedder = TransactionEmbedder()
retrieval_service = get_retrieval_service()
log = get_logger("api")

# Simple optional auth + rate limiting
ENABLE_AUTH = os.getenv("ENABLE_AUTH", "false").lower() == "true"
SECRET_KEY = os.getenv("SECRET_KEY")
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "100"))
_rate_state = {}


async def require_auth(request: Request):
    if not ENABLE_AUTH or not SECRET_KEY:
        return
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    token = auth.split(" ", 1)[1]
    if token != SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid token")


async def rate_limit(request: Request):
    # naive in-memory per-IP limiter: 100 req/min default
    ip = request.client.host if request.client else "unknown"
    from time import time
    now = int(time() // 60)  # minute window
    key = (ip, now)
    cnt = _rate_state.get(key, 0) + 1
    _rate_state[key] = cnt
    # cleanup previous windows occasionally
    if len(_rate_state) > 5000:
        for k in list(_rate_state.keys()):
            if k[1] < now:
                _rate_state.pop(k, None)
    if cnt > RATE_LIMIT_PER_MIN:
        raise HTTPException(status_code=429, detail="Rate limit exceeded")
log = get_logger("api")

@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "LumaFin LHAS",
        "version": "0.1.0"
    }

@app.get("/health")
async def health_check(db: Session = Depends(get_db)):
    """Detailed health check including database"""
    try:
        # Test database connection
        db.execute(text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        db_status = f"error: {str(e)}"

    return {
        "status": "ok",
        "database": db_status,
        "embedder_dimension": embedder.dimension,
        "rules_loaded": len(rule_engine.get_all_rules())
    }

@app.post("/categorize", response_model=Prediction)
async def categorize_transaction(
    transaction: Transaction,
    db: Session = Depends(get_db),
    _auth: None = Depends(require_auth),
    _rl: None = Depends(rate_limit),
):
    """
    Categorize a single transaction using hierarchical decision pipeline.

    Priority order:
    1. Rule engine (confidence = 1.0)
    2. Personal centroids (confidence = 0.85-0.95)
    3. Global retrieval + reranker (confidence = varies)
    4. Fallback to "Uncategorized" (confidence < 0.50)
    """
    try:
        # Compute time features if not provided
        hour = getattr(transaction, 'hour_of_day', None)
        weekday = getattr(transaction, 'weekday', None)
        if hour is None or weekday is None:
            time_feats = get_time_features(transaction.txn_date)
            hour = time_feats.get('hour_of_day') or time_feats.get('hour')
            weekday = time_feats.get('weekday') or time_feats.get('day_of_week')
        # Normalize user-provided label if present
        label = transaction.label
        if label:
            label = normalize_label(label)

        cat, conf, expl_dict = decide({
            "merchant": transaction.merchant,
            "amount": float(transaction.amount),
            "description": transaction.description,
            "user_id": transaction.user_id,
            "label": label,
            "hour_of_day": hour,
            "weekday": weekday,
        })
        explanation = build_explanation(
            decision_path=expl_dict.get("decision_path", "fallback"),
            nearest_examples=expl_dict.get("nearest_examples"),
            rule_matched=expl_dict.get("rule_matched"),
            centroid_similarity=expl_dict.get("centroid_similarity"),
            shap_values=expl_dict.get("feature_importance"),
        )
        category = Category(name=cat, confidence=conf, is_personal=expl_dict.get("decision_path") == "centroid")
        pred = Prediction(transaction=transaction, category=category, explanation=explanation)
        # Attach label influence if present in explanation
        pred.predicted_label = expl_dict.get("predicted_label")
        pred.label_influence = expl_dict.get("label_influence")
        return pred
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Categorization failed: {str(e)}")

@app.get("/taxonomy/global")
async def get_global_taxonomy(db: Session = Depends(get_db)):
    """Get list of global categories"""
    result = db.execute(text("SELECT category_name FROM global_taxonomy ORDER BY category_name"))
    categories = [row[0] for row in result]
    return {"categories": categories}

@app.get("/rules")
async def get_rules():
    """Get all loaded rules for debugging"""
    return {"rules": rule_engine.get_all_rules()}

@app.post("/rules/refresh")
async def refresh_rules():
    """Refresh rules from database and YAML"""
    rule_engine.refresh_rules()
    return {"message": "Rules refreshed", "count": len(rule_engine.get_all_rules())}


@app.get("/taxonomy/personal/{user_id}")
async def get_personal_taxonomy(user_id: int, db: Session = Depends(get_db)):
    result = db.execute(text("""
        SELECT category_name, num_transactions, metadata
        FROM personal_centroids
        WHERE user_id = :uid
        ORDER BY num_transactions DESC
    """), {"uid": user_id})
    items = []
    for row in result:
        items.append({
            "category": row[0],
            "num_transactions": row[1],
            "metadata": row[2],
        })
    return {"user_id": user_id, "categories": items}


@app.post("/taxonomy/retrain/{user_id}")
async def retrain_personal_taxonomy(user_id: int):
    engine = AMPTClusteringEngine()
    clusters = engine.cluster_user(user_id)
    return {"user_id": user_id, "clusters": len(clusters)}


@app.post("/feedback")
async def submit_feedback(feedback: dict, db: Session = Depends(get_db)):
    """Submit user correction feedback.

    Expected payload keys: user_id, transaction_id, predicted_category, correct_category
    """
    try:
        db.execute(text("""
            INSERT INTO feedback_queue (user_id, transaction_id, predicted_category, correct_category, user_label, hour_of_day, weekday)
            VALUES (:uid, :tid, :pred, :corr, :label, :hour, :weekday)
        """), {
            "uid": feedback.get("user_id"),
            "tid": feedback.get("transaction_id"),
            "pred": feedback.get("predicted_category"),
            "corr": feedback.get("correct_category"),
            "label": feedback.get("user_label"),
            "hour": feedback.get("hour_of_day"),
            "weekday": feedback.get("weekday"),
        })
        db.commit()
        return {"status": "queued"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feedback failed: {str(e)}")