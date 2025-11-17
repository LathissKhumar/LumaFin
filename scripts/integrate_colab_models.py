#!/usr/bin/env python3
"""
Script to integrate Google Colab trained models into local LumaFin repository.

This script helps you:
1. Verify downloaded model files
2. Move them to correct locations
3. Update .env configuration
4. Test model loading

Usage:
    python scripts/integrate_colab_models.py --source /path/to/downloaded/models
"""

import argparse
import os
import shutil
import sys
from pathlib import Path


def check_file_exists(path: str, description: str) -> bool:
    """Check if a file exists and print status."""
    if os.path.exists(path):
        size_mb = os.path.getsize(path) / (1024 * 1024)
        print(f"  ✅ {description}: {path} ({size_mb:.1f} MB)")
        return True
    else:
        print(f"  ❌ {description}: NOT FOUND at {path}")
        return False


def check_directory_exists(path: str, description: str) -> bool:
    """Check if a directory exists and print status."""
    if os.path.isdir(path):
        # Count files recursively
        file_count = sum(1 for _ in Path(path).rglob('*') if _.is_file())
        print(f"  ✅ {description}: {path} ({file_count} files)")
        return True
    else:
        print(f"  ❌ {description}: NOT FOUND at {path}")
        return False


def verify_models(source_dir: str) -> dict:
    """Verify that all required model files are present."""
    print("\n🔍 Verifying model files...\n")
    
    results = {}
    
    # Check embedding model
    embedding_dir = os.path.join(source_dir, "lumafin-lacft-v1.0")
    results['embedding'] = check_directory_exists(embedding_dir, "Embedding model")
    
    # Check reranker models
    reranker_pkl = os.path.join(source_dir, "xgb_reranker.pkl")
    reranker_json = os.path.join(source_dir, "xgb_reranker.json")
    results['reranker_pkl'] = check_file_exists(reranker_pkl, "Reranker (PKL)")
    results['reranker_json'] = check_file_exists(reranker_json, "Reranker (JSON)")
    
    # Check FAISS index
    faiss_index = os.path.join(source_dir, "faiss_index.bin")
    results['faiss_index'] = check_file_exists(faiss_index, "FAISS index")
    
    # Check FAISS metadata
    faiss_metadata = os.path.join(source_dir, "faiss_metadata.pkl")
    results['faiss_metadata'] = check_file_exists(faiss_metadata, "FAISS metadata")
    
    print("\n" + "="*60)
    success_count = sum(1 for v in results.values() if v)
    total_count = len(results)
    print(f"Verification: {success_count}/{total_count} items found")
    print("="*60 + "\n")
    
    return results


def copy_models(source_dir: str, target_dir: str, verification: dict) -> bool:
    """Copy models to target directory."""
    print("📦 Copying models to repository...\n")
    
    # Create target directories
    embeddings_target = os.path.join(target_dir, "models", "embeddings")
    reranker_target = os.path.join(target_dir, "models", "reranker")
    os.makedirs(embeddings_target, exist_ok=True)
    os.makedirs(reranker_target, exist_ok=True)
    
    success = True
    
    # Copy embedding model
    if verification['embedding']:
        src = os.path.join(source_dir, "lumafin-lacft-v1.0")
        dst = os.path.join(embeddings_target, "lumafin-lacft-v1.0")
        if os.path.exists(dst):
            print(f"  ⚠️  Removing existing embedding model at {dst}")
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        print(f"  ✅ Copied embedding model to {dst}")
    else:
        print("  ⚠️  Skipping embedding model (not found)")
        success = False
    
    # Copy reranker (prefer JSON format)
    if verification['reranker_json']:
        src = os.path.join(source_dir, "xgb_reranker.json")
        dst = os.path.join(reranker_target, "xgb_reranker.json")
        shutil.copy2(src, dst)
        print(f"  ✅ Copied reranker (JSON) to {dst}")
    elif verification['reranker_pkl']:
        src = os.path.join(source_dir, "xgb_reranker.pkl")
        dst = os.path.join(reranker_target, "xgb_reranker.pkl")
        shutil.copy2(src, dst)
        print(f"  ✅ Copied reranker (PKL) to {dst}")
    else:
        print("  ⚠️  Skipping reranker (not found)")
        success = False
    
    # Copy FAISS index
    if verification['faiss_index']:
        src = os.path.join(source_dir, "faiss_index.bin")
        dst = os.path.join(target_dir, "models", "faiss_index.bin")
        shutil.copy2(src, dst)
        print(f"  ✅ Copied FAISS index to {dst}")
    else:
        print("  ⚠️  Skipping FAISS index (not found)")
        success = False
    
    # Copy FAISS metadata
    if verification['faiss_metadata']:
        src = os.path.join(source_dir, "faiss_metadata.pkl")
        dst = os.path.join(target_dir, "models", "faiss_metadata.pkl")
        shutil.copy2(src, dst)
        print(f"  ✅ Copied FAISS metadata to {dst}")
    else:
        print("  ⚠️  Skipping FAISS metadata (not found)")
        success = False
    
    print()
    return success


def update_env_file(target_dir: str, verification: dict):
    """Update or create .env file with model paths."""
    print("⚙️  Updating .env configuration...\n")
    
    env_file = os.path.join(target_dir, ".env")
    env_example = os.path.join(target_dir, ".env.example")
    
    # Read existing .env or use .env.example as template
    if os.path.exists(env_file):
        with open(env_file, 'r') as f:
            env_content = f.read()
    elif os.path.exists(env_example):
        with open(env_example, 'r') as f:
            env_content = f.read()
    else:
        env_content = ""
    
    # Update or add model paths
    updates = []
    
    if verification['embedding']:
        if 'MODEL_PATH=' in env_content:
            env_content = env_content.replace(
                'MODEL_PATH=sentence-transformers/all-MiniLM-L6-v2',
                'MODEL_PATH=models/embeddings/lumafin-lacft-v1.0'
            )
        else:
            updates.append('MODEL_PATH=models/embeddings/lumafin-lacft-v1.0')
    
    if verification['reranker_json']:
        if 'RERANKER_MODEL_PATH=' not in env_content:
            updates.append('RERANKER_MODEL_PATH=models/reranker/xgb_reranker.json')
    
    if verification['faiss_index']:
        if 'FAISS_INDEX_PATH=' not in env_content:
            updates.append('FAISS_INDEX_PATH=models/faiss_index.bin')
    
    if verification['faiss_metadata']:
        if 'FAISS_METADATA_PATH=' not in env_content:
            updates.append('FAISS_METADATA_PATH=models/faiss_metadata.pkl')
    
    # Append new configurations
    if updates:
        if env_content and not env_content.endswith('\n'):
            env_content += '\n'
        env_content += '\n# Colab-trained models\n'
        env_content += '\n'.join(updates) + '\n'
    
    # Write back
    with open(env_file, 'w') as f:
        f.write(env_content)
    
    print(f"  ✅ Updated {env_file}")
    print("\n  Configuration added:")
    for update in updates:
        print(f"    {update}")
    print()


def test_models(target_dir: str, verification: dict):
    """Test loading the models."""
    print("🧪 Testing model loading...\n")
    
    # Add target directory to path
    sys.path.insert(0, target_dir)
    
    try:
        # Test embedding model
        if verification['embedding']:
            from sentence_transformers import SentenceTransformer
            model_path = os.path.join(target_dir, "models", "embeddings", "lumafin-lacft-v1.0")
            model = SentenceTransformer(model_path)
            embedding = model.encode("test transaction")
            print(f"  ✅ Embedding model loaded successfully (dim: {len(embedding)})")
        
        # Test reranker
        if verification['reranker_json']:
            import xgboost as xgb
            model_path = os.path.join(target_dir, "models", "reranker", "xgb_reranker.json")
            model = xgb.XGBClassifier()
            model.load_model(model_path)
            print(f"  ✅ Reranker model loaded successfully")
        
        # Test FAISS index
        if verification['faiss_index']:
            import faiss
            import pickle
            
            index_path = os.path.join(target_dir, "models", "faiss_index.bin")
            index = faiss.read_index(index_path)
            print(f"  ✅ FAISS index loaded successfully ({index.ntotal} vectors)")
            
            if verification['faiss_metadata']:
                metadata_path = os.path.join(target_dir, "models", "faiss_metadata.pkl")
                with open(metadata_path, 'rb') as f:
                    metadata = pickle.load(f)
                print(f"  ✅ FAISS metadata loaded ({len(metadata.get('categories', []))} categories)")
        
        print("\n✅ All models loaded successfully!\n")
        return True
        
    except Exception as e:
        print(f"\n❌ Error loading models: {e}\n")
        print("Make sure you have installed all dependencies:")
        print("  pip install sentence-transformers xgboost faiss-cpu")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Integrate Google Colab trained models into LumaFin repository"
    )
    parser.add_argument(
        '--source',
        type=str,
        required=True,
        help='Path to downloaded models directory from Google Drive'
    )
    parser.add_argument(
        '--target',
        type=str,
        default='.',
        help='Path to LumaFin repository (default: current directory)'
    )
    parser.add_argument(
        '--skip-test',
        action='store_true',
        help='Skip model loading tests'
    )
    
    args = parser.parse_args()
    
    # Resolve paths
    source_dir = os.path.abspath(args.source)
    target_dir = os.path.abspath(args.target)
    
    print("\n" + "="*60)
    print("LumaFin - Colab Model Integration")
    print("="*60)
    print(f"Source: {source_dir}")
    print(f"Target: {target_dir}")
    print("="*60)
    
    # Verify source directory exists
    if not os.path.isdir(source_dir):
        print(f"\n❌ Error: Source directory not found: {source_dir}")
        print("Please provide the correct path to your downloaded models.")
        sys.exit(1)
    
    # Verify target is a LumaFin repository
    if not os.path.exists(os.path.join(target_dir, "pyproject.toml")):
        print(f"\n⚠️  Warning: Target directory may not be a LumaFin repository")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(0)
    
    # Step 1: Verify models
    verification = verify_models(source_dir)
    
    if not any(verification.values()):
        print("❌ No models found in source directory. Please check the path.")
        sys.exit(1)
    
    # Step 2: Copy models
    print("Proceed with copying models? (y/n): ", end='')
    if input().lower() != 'y':
        print("Aborted.")
        sys.exit(0)
    
    success = copy_models(source_dir, target_dir, verification)
    
    # Step 3: Update .env
    update_env_file(target_dir, verification)
    
    # Step 4: Test models
    if not args.skip_test and success:
        test_models(target_dir, verification)
    
    print("="*60)
    print("✅ Integration complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Review the .env file to ensure paths are correct")
    print("2. Test the API: PYTHONPATH=. uvicorn src.api.main:app --reload")
    print("3. Try categorizing a transaction via API or Streamlit UI")
    print()


if __name__ == "__main__":
    main()
