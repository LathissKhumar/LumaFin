"""Run a local L-A CFT training job using a small CSV or DB seed examples.

Example usage:
  PYTHONPATH=. python scripts/run_lacft_local.py --epochs 1 --batch-size 8 --csv data/merged_training.csv
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--csv', default=None)
    args = parser.parse_args()

    cmd = [sys.executable, 'src/embedder/train_lacft.py', '--epochs', str(args.epochs), '--batch-size', str(args.batch_size), '--patience', '1']
    if args.csv:
        cmd += ['--csv', args.csv]

    print('Executing:', ' '.join(cmd))
    subprocess.run(cmd, check=True)

if __name__ == '__main__':
    main()
