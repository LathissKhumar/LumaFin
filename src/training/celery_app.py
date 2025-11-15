"""Celery beat configuration for scheduled tasks.

This configures periodic tasks like:
- Processing feedback queue
- Rebuilding FAISS index
- Re-clustering user centroids

Usage:
  # Start worker
  celery -A src.training.celery_app worker --loglevel=info
  
  # Start beat scheduler
  celery -A src.training.celery_app beat --loglevel=info
"""
from __future__ import annotations

import os
from celery import Celery
from celery.schedules import crontab

# Redis URL from environment
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Create Celery app
app = Celery(
    "lumafin",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=["src.training.feedback_worker", "src.training.incremental"]
)

# Configuration
app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_time_limit=30 * 60,  # 30 minutes max
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
)

# Scheduled tasks
app.conf.beat_schedule = {
    # Process feedback every 5 minutes
    "process-feedback-queue": {
        "task": "src.training.feedback_worker.process_feedback_batch",
        "schedule": crontab(minute="*/5"),  # Every 5 minutes
        "options": {"queue": "feedback"},
    },
    
    # Rebuild FAISS index nightly at 2 AM
    "rebuild-faiss-index": {
        "task": "src.training.incremental.rebuild_faiss_index",
        "schedule": crontab(hour=2, minute=0),  # 2:00 AM daily
        "options": {"queue": "training"},
    },
    
    # Check for users needing re-clustering (daily at 3 AM)
    "check-recluster-users": {
        "task": "src.training.feedback_worker.check_recluster_users",
        "schedule": crontab(hour=3, minute=0),  # 3:00 AM daily
        "options": {"queue": "clustering"},
    },
}

# Task routing
app.conf.task_routes = {
    "src.training.feedback_worker.*": {"queue": "feedback"},
    "src.training.incremental.*": {"queue": "training"},
}

# Environment-specific overrides
if os.getenv("CELERY_DISABLE_BEAT") == "true":
    app.conf.beat_schedule = {}
    
if os.getenv("FEEDBACK_INTERVAL_MINUTES"):
    interval = int(os.getenv("FEEDBACK_INTERVAL_MINUTES"))
    app.conf.beat_schedule["process-feedback-queue"]["schedule"] = crontab(minute=f"*/{interval}")


__all__ = ["app"]
