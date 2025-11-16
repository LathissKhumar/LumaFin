"""Build FAISS index from database using the embedder and persist index/metadata.

This script reads global_examples from the DB, ensures embeddings exist using
the TransactionEmbedder (L-A CFT), builds a FAISS IndexFlatIP, and writes it to
disk in `models/faiss_index.bin` and metadata to `models/faiss_metadata.pkl`.
"""
from __future__ import annotations

import os
import argparse
import numpy as np
from sqlalchemy import text
from src.indexer.faiss_builder import FAISSIndexBuilder
from src.storage.database import SessionLocal


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index-path', default='models/faiss_index.bin')
    parser.add_argument('--metadata-path', default='models/faiss_metadata.pkl')
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    print('Loading examples from DB...')
    db = SessionLocal()
    try:
        q = text('SELECT ge.id, ge.merchant, ge.amount, ge.description, gt.category_name, ge.embedding FROM global_examples ge JOIN global_taxonomy gt ON ge.category_id = gt.id')
        if args.limit:
            q = text(q.text + ' LIMIT :lim')
            rows = db.execute(q, {'lim': args.limit})
        else:
            rows = db.execute(q)
        examples = []
        data = []
        for row in rows:
            examples.append({'id': row[0], 'merchant': row[1], 'amount': float(row[2]) if row[2] else 0.0, 'description': row[3], 'category': row[4]})
            if row[5] is not None:
                emb = np.array(eval(row[5]), dtype=np.float32)
                data.append(emb)
            else:
                data.append(None)
    finally:
        db.close()

    # If the DB returned no rows, try fallback to CSV source
    if len(examples) == 0:
        print('[build_faiss] Warning: No DB examples found, falling back to `data/merged_training.csv`')
        import csv
        csv_path = 'data/merged_training.csv'
        if os.path.exists(csv_path):
            with open(csv_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                examples = []
                data = []
                for i, row in enumerate(reader):
                    merchant = row.get('merchant') or row.get('text') or row.get('description') or ''
                    category = row.get('category') or row.get('label') or 'Uncategorized'
                    amount = float(row.get('amount') or 0.0)
                    examples.append({'id': i, 'merchant': merchant, 'amount': amount, 'description': row.get('description', ''), 'category': category})
                    data.append(None)
        else:
            print('[build_faiss] Error: No DB rows and CSV fallback not found. Aborting FAISS build.')
            return 1

    # Use embedder to fill missing embeddings
    from src.embedder.encoder import TransactionEmbedder
    embd = TransactionEmbedder()
    embeddings = []
    for ex, emb in zip(examples, data):
        if emb is None:
            v = embd.encode_transaction(ex['merchant'], ex['amount'], ex['description'])
        else:
            v = emb
        embeddings.append(np.array(v, dtype=np.float32))

    print(f'Preparing to build FAISS index with {len(embeddings)} vectors')
    if len(embeddings) == 0:
        print('[build_faiss] Error: No embeddings available to build FAISS index. Aborting.')
        return 1
    builder = FAISSIndexBuilder(embd)
    idx = builder.build_index(embeddings, examples)
    os.makedirs(os.path.dirname(args.index_path), exist_ok=True)
    builder.save(args.index_path, args.metadata_path)
    print('FAISS index built and saved.')


if __name__ == '__main__':
    main()
