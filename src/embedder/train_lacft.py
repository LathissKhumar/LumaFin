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
import csv
from dataclasses import dataclass
from typing import List, Dict, Sequence, Optional

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

from sentence_transformers import SentenceTransformer
from sqlalchemy import text

from src.storage.database import SessionLocal
from src.preprocessing.normalize import normalize_merchant, bucket_amount

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@dataclass
class LabeledExample:
    text: str
    category: str


class ContrastiveDataset(Dataset):
    """Contrastive dataset with optional hard negative mining.

    If hard_negative_mining is enabled we will, for each anchor, try to pick a
    negative example whose text has high lexical similarity (shared tokens) but
    a different category – these are harder for the model and accelerate learning.
    """
    def __init__(
        self,
        examples: List[LabeledExample],
        hard_negative_mining: bool = False,
        max_hard_negative_trials: int = 15,
    ):
        self.examples = examples
        self.hard_negative_mining = hard_negative_mining
        self.max_hard_negative_trials = max_hard_negative_trials
        # Group indices by category for positive sampling
        self.by_category: Dict[str, List[int]] = {}
        for idx, ex in enumerate(examples):
            self.by_category.setdefault(ex.category, []).append(idx)
        self.categories = list(self.by_category.keys())
        # Pre-tokenize for cheap lexical similarity scoring
        self._tokens: List[set] = [set(e.text.split()) for e in self.examples]

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
            positive = anchor  # fallback

        # Negative selection
        if not self.hard_negative_mining:
            neg_cat = random.choice([c for c in self.categories if c != anchor.category])
            neg_idx = random.choice(self.by_category[neg_cat])
            negative = self.examples[neg_idx]
        else:
            # Attempt to find lexically similar but different-category negative
            anchor_tokens = self._tokens[idx]
            best_idx: Optional[int] = None
            best_overlap = -1
            trials = 0
            while trials < self.max_hard_negative_trials:
                trials += 1
                neg_cat = random.choice([c for c in self.categories if c != anchor.category])
                candidate_idx = random.choice(self.by_category[neg_cat])
                overlap = len(anchor_tokens & self._tokens[candidate_idx])
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = candidate_idx
            if best_idx is None:
                # Fallback random negative
                neg_cat = random.choice([c for c in self.categories if c != anchor.category])
                best_idx = random.choice(self.by_category[neg_cat])
            negative = self.examples[best_idx]

        return anchor.text, positive.text, negative.text


class LabelAwareContrastiveTrainer:
    def __init__(self, model_name: str, lr: float = 2e-5, temperature: float = 0.07):
        self.model = SentenceTransformer(model_name)
        self.model.to(DEVICE)
        self.temperature = temperature
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr)

    def encode(self, texts: List[str]) -> torch.Tensor:
        # Tokenize manually to maintain gradients
        features = self.model.tokenize(texts)
        features = {k: v.to(DEVICE) for k, v in features.items()}
        # Forward pass through the model
        output = self.model.forward(features)
        # Get sentence embeddings (mean pooling)
        embeddings = output['sentence_embedding']
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
            self.optimizer.zero_grad()
            anchor_texts, pos_texts, neg_texts = batch
            anchor_emb = self.encode(anchor_texts)
            pos_emb = self.encode(pos_texts)
            neg_emb = self.encode(neg_texts)

            loss = self.contrastive_loss(anchor_emb, pos_emb, neg_emb)
            loss.backward()
            self.optimizer.step()

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
            text_val = f"{normalized} {bucket_amount(float(amount))}" if amount is not None else normalized
            examples.append(LabeledExample(text=text_val, category=category))
        return examples
    finally:
        db.close()


def load_csv_examples(paths: Sequence[str]) -> List[LabeledExample]:
    """Load labeled examples from one or more CSV files.

    Accepted column names (case-insensitive): merchant|text, category|label, amount(optional), description(optional)
    """
    out: List[LabeledExample] = []
    for p in paths:
        if not p or not os.path.exists(p):
            print(f"[load_csv_examples] Skipping missing path: {p}")
            continue
        with open(p, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = {c.lower(): c for c in reader.fieldnames or []}
            merchant_col = cols.get("merchant") or cols.get("text") or cols.get("description")
            category_col = cols.get("category") or cols.get("label")
            amount_col = cols.get("amount")
            if not merchant_col or not category_col:
                print(f"[load_csv_examples] Required columns not found in {p}; skipping.")
                continue
            for row in reader:
                merchant_raw = row.get(merchant_col, "").strip()
                category_raw = row.get(category_col, "").strip() or "Uncategorized"
                if not merchant_raw:
                    continue
                normalized = normalize_merchant(merchant_raw)
                amt_val = None
                if amount_col and row.get(amount_col):
                    try:
                        amt_val = float(row.get(amount_col))
                    except Exception:
                        amt_val = None
                text = normalized
                if amt_val is not None:
                    text = f"{normalized} {bucket_amount(amt_val)}"
                out.append(LabeledExample(text=text, category=category_raw))
        print(f"Loaded {sum(1 for _ in out)} cumulative examples after {p}")
    return out


def balance_classes(examples: List[LabeledExample], target: Optional[int] = None) -> List[LabeledExample]:
    """Simple oversampling to balance class counts.
    target: if provided use this per-class count else use median.
    """
    by_cat: Dict[str, List[LabeledExample]] = {}
    for ex in examples:
        by_cat.setdefault(ex.category, []).append(ex)
    sizes = [len(v) for v in by_cat.values()]
    if not sizes:
        return examples
    per_class = target or int(sorted(sizes)[len(sizes)//2])
    balanced: List[LabeledExample] = []
    for cat, items in by_cat.items():
        if len(items) >= per_class:
            balanced.extend(random.sample(items, per_class))
        else:
            # Oversample with replacement
            needed = per_class - len(items)
            balanced.extend(items + [random.choice(items) for _ in range(needed)])
    random.shuffle(balanced)
    print(f"Balanced dataset size: {len(balanced)} (per-class ~{per_class})")
    return balanced


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--output-dir", default="models/embeddings/lumafin-lacft-v1.0")
    parser.add_argument("--csv", nargs="*", help="Optional one or more CSV files to augment training data")
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--hard-negatives", action="store_true", help="Enable lexical hard negative mining")
    parser.add_argument("--balance", action="store_true", help="Balance classes via oversampling")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience (epochs)")
    args = parser.parse_args()

    print("Loading seed examples from DB...")
    examples = load_seed_examples()
    print(f"Loaded {len(examples)} seed examples")

    if args.csv:
        csv_ex = load_csv_examples(args.csv)
        print(f"Loaded {len(csv_ex)} CSV examples")
        examples.extend(csv_ex)
        print(f"Combined dataset size: {len(examples)}")

    if args.balance:
        examples = balance_classes(examples)

    dataset = ContrastiveDataset(examples, hard_negative_mining=args.hard_negatives)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    trainer = LabelAwareContrastiveTrainer(args.model_name, lr=args.lr, temperature=args.temperature)

    best_loss = float("inf")
    epochs_no_improve = 0
    for epoch in range(1, args.epochs + 1):
        avg_loss = trainer.train_epoch(dataloader, epoch)
        if avg_loss < best_loss - 1e-4:
            best_loss = avg_loss
            epochs_no_improve = 0
            trainer.save(args.output_dir)
            print(f"Improved loss {avg_loss:.4f}; model checkpoint saved.")
        else:
            epochs_no_improve += 1
            print(f"No improvement this epoch (avg {avg_loss:.4f}); {epochs_no_improve}/{args.patience} without improvement.")
            if epochs_no_improve >= args.patience:
                print("Early stopping triggered.")
                break

    print("Training complete.")


if __name__ == "__main__":
    main()
