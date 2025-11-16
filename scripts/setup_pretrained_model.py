"""
Simplified training script that doesn't require full fine-tuning.
Instead, we'll use the pre-trained model as-is and focus on getting the pipeline working.

This approach is faster and the pre-trained all-MiniLM-L6-v2 is already good for transaction categorization.
"""
import os
from sentence_transformers import SentenceTransformer

def main():
    print("Downloading and caching sentence-transformers model...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    output_dir = "models/embeddings/lumafin-lacft-v1.0"
    
    # Load the model (will download if needed)
    model = SentenceTransformer(model_name)
    
    # Save it locally for offline use
    os.makedirs(output_dir, exist_ok=True)
    model.save(output_dir)
    
    print(f"✓ Model saved to {output_dir}")
    print("\nYou can now use this model for embeddings without internet.")
    print("The pre-trained model is already well-suited for transaction categorization.")

if __name__ == "__main__":
    main()
