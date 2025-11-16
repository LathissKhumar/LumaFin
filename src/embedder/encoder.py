import os
from typing import List, Dict

import numpy as np
from sentence_transformers import SentenceTransformer

from src.preprocessing.normalize import normalize_merchant, bucket_amount


class TransactionEmbedder:
    """Encode transactions into sentence embeddings."""

    def __init__(self, model_name: str | None = None):
        # Prefer explicit param, then EMBEDDING_MODEL, then MODEL_PATH
        if model_name is None:
            model_name = os.getenv("EMBEDDING_MODEL") or os.getenv("MODEL_PATH") or "sentence-transformers/all-MiniLM-L6-v2"
        # Allow local directory fallback to support offline environments
        try:
            self.model = SentenceTransformer(model_name)
        except Exception as e:
            alt = os.getenv("MODEL_PATH")
            if alt and alt != model_name:
                self.model = SentenceTransformer(alt)
            else:
                raise e
        self.dimension = self.model.get_sentence_embedding_dimension()

    def encode_transaction(self, merchant: str, amount: float | None = None, description: str | None = None, label: str | None = None, hour_of_day: int | None = None, weekday: int | None = None) -> np.ndarray:
        text = normalize_merchant(merchant)
        if amount is not None:
            text += f" {bucket_amount(float(amount))}"
        if description:
            text += f" {description}"
        # Optional user label embedding augmentation
        if label:
            # Keep label short and normalized
            from src.preprocessing.normalize import normalize_label
            nl = normalize_label(label)
            text += f" label:{nl}"
        if hour_of_day is not None:
            text += f" hour:{int(hour_of_day)}"
        if weekday is not None:
            text += f" weekday:{int(weekday)}"
        return self.model.encode(text, convert_to_numpy=True)

    def encode_batch(self, transactions: List[Dict]) -> np.ndarray:
        texts: List[str] = []
        for txn in transactions:
            t = normalize_merchant(txn.get("merchant", ""))
            amt = txn.get("amount")
            if amt is not None:
                t += f" {bucket_amount(float(amt))}"
            desc = txn.get("description")
            if desc:
                t += f" {desc}"
            if txn.get("label"):
                try:
                    from src.preprocessing.normalize import normalize_label
                    t += f" label:{normalize_label(txn.get('label'))}"
                except Exception:
                    pass
            if txn.get("hour_of_day") is not None:
                t += f" hour:{int(txn.get('hour_of_day'))}"
            if txn.get("weekday") is not None:
                t += f" weekday:{int(txn.get('weekday'))}"
            texts.append(t)
        return self.model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
