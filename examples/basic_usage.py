"""
Basic usage example for EmbedCache

This example shows how to use EmbedCache for simple embedding caching.
"""

import numpy as np
from embedcache import EmbedCache


def main():
    print("EmbedCache - Basic Usage Example\n")

    # Create a cache (dimension = 384 for sentence-transformers/all-MiniLM-L6-v2)
    cache = EmbedCache(path="./my_cache.db", dimension=384)

    print(f"Created cache: {cache}\n")

    # Simulate some embeddings
    texts = [
        "Hello world",
        "Machine learning is amazing",
        "Python is a great language",
        "Vector databases are useful",
    ]

    print("Caching embeddings...")
    for text in texts:
        # Simulate embedding (normally you'd call your embedding model here)
        vector = np.random.randn(384).astype(np.float32)
        vector = vector / np.linalg.norm(vector)  # Normalize

        cache.set(text, vector)
        print(f"  Cached: '{text}'")

    print(f"\nCache stats: {cache.stats()}\n")

    # Retrieve embeddings
    print("Retrieving cached embeddings...")
    for text in texts:
        vector = cache.get(text)
        if vector is not None:
            print(f"  Retrieved: '{text}' (dimension: {len(vector)})")
        else:
            print(f"  Not found: '{text}'")

    # Test similarity matching
    print("\nTesting similarity matching...")
    cache_with_similarity = EmbedCache(
        path="./similarity_cache.db",
        dimension=384,
        similarity_threshold=0.95
    )

    # Add some embeddings
    vector1 = np.random.randn(384).astype(np.float32)
    vector1 = vector1 / np.linalg.norm(vector1)
    cache_with_similarity.set("hello world", vector1)

    # Try to find similar
    query = vector1 + np.random.randn(384).astype(np.float32) * 0.01  # Slightly perturbed
    query = query / np.linalg.norm(query)

    result = cache_with_similarity.find_similar(query, threshold=0.90)
    if result:
        cached_vector, score = result
        print(f"  Found similar embedding with score: {score:.4f}")
    else:
        print("  No similar embedding found")

    # Clean up
    cache.close()
    cache_with_similarity.close()

    print("\nDone!")


if __name__ == "__main__":
    main()
