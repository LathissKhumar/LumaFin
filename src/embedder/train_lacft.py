"""Label-Aware Contrastive Fine-Tuning (L-A CFT) Trainer

This is a simplified contrastive training loop that:
- Loads seed labeled examples from the database (merchant, category)
- Generates anchor/positive/negative pairs
- Optimizes embedding model so same-category pairs are closer than different ones.

Run:
    PYTHONPATH=. python src/embedder/train_lacft.py --epochs 1 --batch-size 32

NOTE: This is a minimal implementation for learning purposes. In production you'd:
- Use more robust sampling (hard negatives)
- Mixed precision
- Gradient accumulation for large batches
- Proper evaluation & checkpointing
"""
from __future__ import annotations

import argparse
import random
import os
from dataclasses import dataclass
from typing import List, Dict

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sqlalchemy import text

from sentence_transformers import SentenceTransformer

from src.storage.database import SessionLocal
from src.preprocessing.normalize import normalize_merchant, bucket_amount

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class LabeledExample:
    text: str
    category: str


class ContrastiveDataset(Dataset):
    def __init__(self, examples: List[LabeledExample]):
        self.examples = examples
        # Group indices by category for positive sampling
        self.by_category: Dict[str, List[int]] = {}
        for idx, ex in enumerate(examples):
            self.by_category.setdefault(ex.category, []).append(idx)
        self.categories = list(self.by_category.keys())

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx: int):
        anchor = self.examples[idx]
        # Positive: random other example from same category (if available)
        pos_pool = self.by_category[anchor.category]
        if len(pos_pool) > 1:
            pos_idx = random.choice([i for i in pos_pool if i != idx])
            positive = self.examples[pos_idx]
        else:
            positive = anchor  # fallback (will not contribute much)

        # Negative: random example from different category
        neg_cat = random.choice([c for c in self.categories if c != anchor.category])
        neg_idx = random.choice(self.by_category[neg_cat])
        negative = self.examples[neg_idx]

        return anchor.text, positive.text, negative.text


class LabelAwareContrastiveTrainer:
    def __init__(self, model_name: str, lr: float = 2e-5, temperature: float = 0.07):
        self.model = SentenceTransformer(model_name)
        self.temperature = temperature
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def encode(self, texts: List[str]) -> torch.Tensor:
        embeddings = self.model.encode(texts, convert_to_tensor=True, show_progress_bar=False)
        return embeddings  # shape (batch, dim)

    def contrastive_loss(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        # Normalize to unit vectors
        anchor = nn.functional.normalize(anchor, dim=-1)
        positive = nn.functional.normalize(positive, dim=-1)
        negative = nn.functional.normalize(negative, dim=-1)

        # Cosine similarities
        pos_sim = (anchor * positive).sum(dim=-1)  # shape (batch,)
        neg_sim = (anchor * negative).sum(dim=-1)  # shape (batch,)

        # Loss: maximize pos_sim, minimize neg_sim
        logits = torch.stack([pos_sim, neg_sim], dim=1)  # (batch, 2)
        labels = torch.zeros(anchor.size(0), dtype=torch.long, device=anchor.device)
        loss = nn.functional.cross_entropy(logits / self.temperature, labels)
        return loss

    def train_epoch(self, dataloader: DataLoader, epoch: int):
        self.model.train()
        total_loss = 0.0
        for step, batch in enumerate(dataloader):
            anchor_texts, pos_texts, neg_texts = batch
            anchor_emb = self.encode(anchor_texts)
            pos_emb = self.encode(pos_texts)
            neg_emb = self.encode(neg_texts)

            loss = self.contrastive_loss(anchor_emb, pos_emb, neg_emb)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

            total_loss += loss.item()
            if (step + 1) % 10 == 0:
                print(f"Epoch {epoch} Step {step+1}/{len(dataloader)} Loss: {loss.item():.4f}")
        avg = total_loss / len(dataloader)
        print(f"Epoch {epoch} average loss: {avg:.4f}")
        return avg

    def save(self, output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        self.model.save(output_dir)
        print(f"Saved fine-tuned model to {output_dir}")


def load_seed_examples(limit: int = 500) -> List[LabeledExample]:
    db = SessionLocal()
    try:
        result = db.execute(text("""
            SELECT merchant, amount, gt.category_name
            FROM global_examples ge
            JOIN global_taxonomy gt ON ge.category_id = gt.id
            LIMIT :lim
        """), {"lim": limit})
        examples: List[LabeledExample] = []
        for row in result:
            merchant, amount, category = row
            normalized = normalize_merchant(merchant)
            text = f"{normalized} {bucket_amount(float(amount))}" if amount is not None else normalized
            examples.append(LabeledExample(text=text, category=category))
        return examples
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", default="models/embeddings/lumafin-lacft-v1.0")
    args = parser.parse_args()

    print("Loading seed examples from DB...")
    examples = load_seed_examples()
    print(f"Loaded {len(examples)} examples")

    dataset = ContrastiveDataset(examples)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    trainer = LabelAwareContrastiveTrainer(args.model_name, lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        trainer.train_epoch(dataloader, epoch)

    trainer.save(args.output_dir)
    print("Training complete.")


if __name__ == "__main__":
    main()
