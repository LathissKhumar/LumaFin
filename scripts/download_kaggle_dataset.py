# Download Kaggle dataset for transaction categorization
# Usage: python scripts/download_kaggle_dataset.py --dataset mlg-ulb/creditcardfraud
import argparse
import os
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help='Kaggle dataset slug (e.g. mlg-ulb/creditcardfraud)')
    parser.add_argument('--dest', type=str, default='data/kaggle', help='Destination folder')
    args = parser.parse_args()

    os.makedirs(args.dest, exist_ok=True)
    cmd = [
        'kaggle', 'datasets', 'download', '-d', args.dataset, '-p', args.dest, '--unzip'
    ]
    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print(f"Downloaded and extracted to {args.dest}")

if __name__ == '__main__':
    main()
