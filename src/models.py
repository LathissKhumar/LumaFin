from __future__ import annotations

from datetime import datetime, date
from decimal import Decimal
from typing import Optional, List, Dict

from pydantic import BaseModel, Field


class Transaction(BaseModel):
    """Input transaction for categorization"""
    merchant: str = Field(..., description="Merchant name")
    amount: Decimal = Field(..., description="Transaction amount")
    # Use alias to avoid any potential name/type confusion in some environments
    txn_date: date = Field(..., alias="date", description="Transaction date")
    description: Optional[str] = Field(None, description="Additional description")
    user_id: Optional[int] = Field(None, description="User ID for personalization")
    # Optional user-provided short label (single word or short phrase)
    label: Optional[str] = Field(None, description="User-provided short label for transaction")
    # Optional time-of-payment features (computed from date) if provided server-side
    hour_of_day: Optional[int] = Field(None, description="Hour of payment (0-23)")
    weekday: Optional[int] = Field(None, description="0=Mon..6=Sun")


class Category(BaseModel):
    """Category information"""
    name: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    is_personal: bool = False


class Explanation(BaseModel):
    """Explanation for categorization decision"""
    decision_path: str  # "rule" | "centroid" | "retrieval" | "fallback"
    nearest_examples: Optional[List[Dict]] = None
    rule_matched: Optional[str] = None
    centroid_similarity: Optional[float] = None
    feature_importance: Optional[Dict] = None


class Prediction(BaseModel):
    """Complete categorization result"""
    transaction: Transaction
    category: Category
    explanation: Explanation
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    # Optionally include the label influence score for explainability
    predicted_label: Optional[str] = None
    label_influence: Optional[float] = None


class FeedbackInput(BaseModel):
    """User correction feedback"""
    transaction_id: int
    old_category: str
    new_category: str
    user_id: int

 