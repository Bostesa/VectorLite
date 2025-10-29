"""
OpenAI Embeddings Caching Example

This example shows how to use EmbedCache with OpenAI embeddings to save money.

Note: You'll need to set OPENAI_API_KEY environment variable and install openai package.
"""

import os
from embedcache import EmbedCache


def get_embedding_with_cache(text: str, cache: EmbedCache):
    """
    Get embedding with caching - saves API calls and money!
    """
    def compute_embedding():
        # This would call OpenAI API
        # For this example, we'll just simulate it
        print(f"    [API CALL] Computing embedding for: '{text}'")

        # Uncomment this if you have OpenAI API key:
        # from openai import OpenAI
        # client = OpenAI()
        # response = client.embeddings.create(
        #     input=text,
        #     model="text-embedding-ada-002"
        # )
        # return response.data[0].embedding

        # Simulate OpenAI embedding (1536 dimensions)
        import numpy as np
        vector = np.random.randn(1536).astype(np.float32)
        return vector / np.linalg.norm(vector)

    return cache.get_or_compute(text, compute_embedding, use_similarity=False)


def main():
    print("EmbedCache + OpenAI Example\n")
    print("Saving $$$ on embedding API calls!\n")

    # Create cache for OpenAI ada-002 embeddings (1536 dimensions)
    cache = EmbedCache(
        path="./openai_cache.db",
        dimension=1536,
        similarity_threshold=0.95  # Optional: enable similarity matching
    )

    # Sample texts
    texts = [
        "What is machine learning?",
        "How do I learn Python?",
        "What is a vector database?",
        "What is machine learning?",  # Duplicate - should hit cache!
        "How do I learn Python?",      # Duplicate - should hit cache!
    ]

    print("Getting embeddings (watch for cache hits):\n")

    for i, text in enumerate(texts, 1):
        print(f"{i}. '{text}'")

        # First try to get from cache
        cached = cache.get(text)
        if cached is not None:
            print("    [CACHE HIT] Retrieved from cache!")
            embedding = cached
        else:
            print("    [CACHE MISS] Computing...")
            embedding = get_embedding_with_cache(text, cache)

        print(f"    Embedding dimension: {len(embedding)}\n")

    # Show cache statistics
    stats = cache.stats()
    print(f"\nCache Statistics:")
    print(f"  Total cached embeddings: {stats.records}")
    print(f"  Cache file size: {stats.file_size / 1024:.2f} KB")
    print(f"  Dimension: {stats.dimension}")

    # Calculate savings
    total_texts = len(texts)
    unique_texts = stats.records
    api_calls_saved = total_texts - unique_texts
    cost_per_1k_tokens = 0.0001  # OpenAI ada-002 pricing
    avg_tokens_per_text = 10
    savings = (api_calls_saved * avg_tokens_per_text / 1000) * cost_per_1k_tokens

    print(f"\nSavings:")
    print(f"  API calls saved: {api_calls_saved}")
    print(f"  Estimated cost saved: ${savings:.6f}")
    print(f"  (With 1M cached items, save $100+ per month!)")

    cache.close()
    print("\nDone!")


if __name__ == "__main__":
    main()
