"""
End-to-end test of the categorization pipeline.

Run after seeding database: PYTHONPATH=. python tests/test_integration.py
"""
import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from src.rules.engine import rule_engine
from src.embedder.encoder import TransactionEmbedder
from src.retrieval.service import RetrievalService
from src.indexer.faiss_builder import FAISSIndexBuilder


def test_rule_engine():
    """Test rule-based categorization."""
    print("\n=== Testing Rule Engine ===")
    
    test_cases = [
        ("Netflix", 15.99, "Entertainment"),
        ("Spotify", 9.99, "Entertainment"),
        ("Amazon.com", 45.00, "Shopping"),
        ("Shell Gas", 50.00, "Transportation"),
        ("Whole Foods", 85.00, "Food & Dining"),
    ]
    
    passed = 0
    for merchant, amount, expected in test_cases:
        result = rule_engine.apply_rules(merchant, amount)
        if result and result.name == expected:
            print(f"✓ {merchant} → {result.name}")
            passed += 1
        else:
            actual = result.name if result else "No match"
            print(f"✗ {merchant} → {actual} (expected: {expected})")
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_embedder():
    """Test embedding generation."""
    print("\n=== Testing Embedder ===")
    
    embedder = TransactionEmbedder()
    print(f"Model dimension: {embedder.dimension}")
    
    embedding = embedder.encode_transaction("Starbucks", 5.50)
    print(f"Embedding shape: {embedding.shape}")
    print(f"Embedding sample (first 5): {embedding[:5]}")
    
    # Test batch encoding
    transactions = [
        {"merchant": "Starbucks", "amount": 5.50},
        {"merchant": "Shell", "amount": 45.00},
    ]
    batch_embeddings = embedder.encode_batch(transactions)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")
    
    return embedding.shape[0] == 384


def test_faiss_index():
    """Test FAISS index loading and search."""
    print("\n=== Testing FAISS Index ===")
    
    try:
        builder = FAISSIndexBuilder()
        
        # Try to load saved index
        if os.path.exists('models/faiss_index.bin'):
            builder.load('models/faiss_index.bin', 'models/faiss_metadata.pkl')
            print(f"✓ Loaded index with {len(builder.example_ids)} examples")
            
            # Test search
            embedding = builder.embedder.encode_transaction("Coffee Shop", 6.00)
            results = builder.search(embedding, k=5)
            
            print("\nTop 5 similar transactions for 'Coffee Shop $6.00':")
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result['merchant']} (${result['amount']:.2f}) "
                      f"→ {result['category']} (sim: {result['similarity']:.3f})")
            
            return len(results) == 5
        else:
            print("✗ Index not found. Run: PYTHONPATH=. python scripts/seed_database.py")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def test_retrieval_service():
    """Test retrieval service with vote aggregation."""
    print("\n=== Testing Retrieval Service ===")
    
    try:
        retrieval = RetrievalService()
        
        # Test retrieval
        results = retrieval.retrieve("Coffee House", 7.50, k=10)
        
        if not results:
            print("✗ No results from retrieval (index may not be built)")
            return False
        
        print(f"Found {len(results)} similar transactions")
        
        # Test vote aggregation
        votes = retrieval.get_category_votes(results)
        print(f"\nCategory votes: {votes}")
        
        # Test prediction
        category, confidence = retrieval.predict_from_votes(votes)
        print(f"Predicted: {category} (confidence: {confidence:.3f})")
        
        return category != "Uncategorized"
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


def main():
    print("=" * 60)
    print("LumaFin Integration Tests")
    print("=" * 60)
    
    tests = [
        ("Rule Engine", test_rule_engine),
        ("Embedder", test_embedder),
        ("FAISS Index", test_faiss_index),
        ("Retrieval Service", test_retrieval_service),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n✗ {name} failed with exception: {e}")
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    passed_count = sum(1 for _, passed in results if passed)
    total = len(results)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n{passed_count}/{total} tests passed")
    
    if passed_count == total:
        print("\n✓ All tests passed! System is ready.")
        return 0
    else:
        print("\n✗ Some tests failed. Check output above.")
        return 1


if __name__ == "__main__":
    exit(main())
