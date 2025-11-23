"""
FAISS Index Builder for Global Transaction Examples

Loads global_examples from database, generates embeddings, builds FAISS index.
"""
from __future__ import annotations

import os
import pickle
from typing import List, Tuple, Optional

import faiss
import numpy as np
from sqlalchemy import text
from sqlalchemy.orm import Session

from src.storage.database import SessionLocal
from src.embedder.encoder import TransactionEmbedder


class FAISSIndexBuilder:
    """Build and manage FAISS index for global transaction examples."""

    def __init__(self, embedder: TransactionEmbedder | None = None):
        self.embedder = embedder or TransactionEmbedder()
        self.index: Optional[faiss.Index] = None
        self.example_ids: List[int] = []
        self.category_labels: List[str] = []
        self.merchants: List[str] = []
        self.amounts: List[float] = []
        self.dimension = self.embedder.dimension

    def load_examples_from_db(self, db: Session) -> Tuple[List[dict], List[np.ndarray]]:
        """Load global examples from database."""
        query = text("""
            SELECT ge.id, ge.merchant, ge.amount, ge.description, 
                   gt.category_name, ge.embedding
            FROM global_examples ge
            JOIN global_taxonomy gt ON ge.category_id = gt.id
            ORDER BY ge.id
        """)
        
        result = db.execute(query)
        examples = []
        embeddings = []

        for row in result:
            example = {
                'id': row[0],
                'merchant': row[1],
                'amount': float(row[2]) if row[2] else 0.0,
                'description': row[3],
                'category': row[4]
            }
            examples.append(example)
            
            # If embedding exists in DB, use it; otherwise generate
            if row[5] is not None:
                # pgvector returns embedding as bytes - convert to numpy array
                # Handle both bytes and string representations
                emb_data = row[5]
                if isinstance(emb_data, bytes):
                    emb = np.frombuffer(emb_data, dtype=np.float32)
                elif isinstance(emb_data, str):
                    emb = np.array(eval(emb_data), dtype=np.float32)
                else:
                    # Already a list or array
                    emb = np.array(emb_data, dtype=np.float32)
                embeddings.append(emb)
            else:
                # Generate embedding
                emb = self.embedder.encode_transaction(
                    merchant=example['merchant'],
                    amount=example['amount'],
                    description=example['description']
                )
                embeddings.append(emb)

        return examples, embeddings

    def build_index(self, embeddings: List[np.ndarray], examples: List[dict]) -> faiss.Index:
        """
        Build FAISS IndexFlatIP (inner product for cosine similarity).
        
        Uses L2-normalized vectors so inner product = cosine similarity.
        """
        # Convert to numpy array
        vectors = np.array(embeddings, dtype=np.float32)
        
        # L2 normalize for cosine similarity via inner product
        faiss.normalize_L2(vectors)
        
        # Create Flat IP index (exact search, inner product)
        index = faiss.IndexFlatIP(self.dimension)
        
        # Add vectors
        index.add(vectors)
        
        # Store metadata
        self.example_ids = [ex['id'] for ex in examples]
        self.category_labels = [ex['category'] for ex in examples]
        self.merchants = [ex['merchant'] for ex in examples]
        self.amounts = [ex['amount'] for ex in examples]
        
        self.index = index
        return index

    def search(self, query_embedding: np.ndarray, k: int = 20) -> List[dict]:
        """
        Search for top-k similar examples.
        
        Returns list of dicts with: id, category, merchant, amount, similarity
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Normalize query vector
        query = query_embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(query)
        
        # Search
        scores, indices = self.index.search(query, k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.example_ids):  # Valid index
                results.append({
                    'id': self.example_ids[idx],
                    'category': self.category_labels[idx],
                    'merchant': self.merchants[idx],
                    'amount': self.amounts[idx],
                    'similarity': float(score)
                })
        
        return results

    def save(self, index_path: str, metadata_path: str):
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            raise ValueError("No index to save")
        
        # Save FAISS index
        faiss.write_index(self.index, index_path)
        
        # Save metadata
        metadata = {
            'example_ids': self.example_ids,
            'category_labels': self.category_labels,
            'merchants': self.merchants,
            'amounts': self.amounts,
            'dimension': self.dimension
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
        
        print(f"✓ Saved index to {index_path}")
        print(f"✓ Saved metadata to {metadata_path}")

    def load(self, index_path: str, metadata_path: str):
        """Load FAISS index and metadata from disk."""
        # Load FAISS index
        self.index = faiss.read_index(index_path)
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.example_ids = metadata['example_ids']
        self.category_labels = metadata['category_labels']
        self.merchants = metadata['merchants']
        self.amounts = metadata['amounts']
        self.dimension = metadata['dimension']
        
        print(f"✓ Loaded index from {index_path}")
        print(f"✓ Loaded {len(self.example_ids)} examples")

    def add_vectors(self, embeddings: List[np.ndarray], examples: List[dict]):
        """Add new vectors to existing index."""
        if self.index is None:
            raise ValueError("Index not built. Call build_index() first.")
        
        # Convert and normalize
        vectors = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(vectors)
        
        # Add to index
        self.index.add(vectors)
        
        # Append metadata
        self.example_ids.extend([ex['id'] for ex in examples])
        self.category_labels.extend([ex['category'] for ex in examples])
        self.merchants.extend([ex['merchant'] for ex in examples])
        self.amounts.extend([ex['amount'] for ex in examples])


# Global index instance (lazy loaded)
_global_index: Optional[FAISSIndexBuilder] = None


def get_faiss_index() -> FAISSIndexBuilder:
    """Get or create global FAISS index instance."""
    global _global_index
    if _global_index is None:
        _global_index = FAISSIndexBuilder()
        # Try to load from disk, otherwise will be built on first use
        index_path = "models/faiss_index.bin"
        metadata_path = "models/faiss_metadata.pkl"
        if os.path.exists(index_path) and os.path.exists(metadata_path):
            try:
                _global_index.load(index_path, metadata_path)
            except Exception as e:
                print(f"Warning: Could not load saved index: {e}")
    return _global_index
