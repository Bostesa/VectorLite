"""
Serverless Example - Lambda/Vercel/Cloudflare Workers

Shows how EmbedCache works in serverless with singleton pattern.
Cache persists across warm invocations for 0ms overhead!
"""

import numpy as np
from embedcache import EmbedCache


def lambda_handler(event, context):
    """
    Lambda handler with singleton cache (recommended).

    First invocation: 10ms cold start
    Next 50+ invocations: 0ms (reuses instance from previous call)
    """

    # Singleton pattern - reuses cache across warm invocations
    cache = EmbedCache.for_serverless(
        name="embeddings",
        dimension=1536,
        cache_size=100,  # Keep 100 hot vectors in memory
    )

    text = event.get("text", "Hello world")

    # Check cache (0.001ms if cached, 0.1ms if cold)
    embedding = cache.get(text)
    cache_hit = embedding is not None

    if not cache_hit:
        print(f"Cache MISS: {text}")
        # In production: call OpenAI/Cohere/etc
        embedding = np.random.randn(1536).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        cache.set(text, embedding)
    else:
        print(f"Cache HIT: {text}")

    # Get memory stats
    stats = cache.stats()

    # Don't close! Keep it alive for next invocation
    # Lambda will reuse this instance on warm start

    return {
        "statusCode": 200,
        "body": {
            "embedding": embedding.tolist()[:5],  # First 5 dims for demo
            "cache_hit": cache_hit,
            "memory_mb": stats.memory_usage / 1024 / 1024,
            "cached_vectors": stats.cache_size,
            "total_records": stats.records,
        }
    }


def lambda_handler_manual(event, _context):
    """
    Alternative: Manual cache management (not recommended).

    This creates a new cache instance each time.
    Use for_serverless() instead for better performance.
    """
    cache = EmbedCache(
        path="/tmp/manual.cache",
        dimension=1536,
    )

    text = event.get("text", "Hello world")
    embedding = cache.get(text)

    if embedding is None:
        embedding = np.random.randn(1536).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)
        cache.set(text, embedding)

    cache.close()  # Must close to write index

    return {"statusCode": 200, "embedding": embedding.tolist()[:5]}


def main():
    """Simulate Lambda warm container reusing cache"""
    print("Simulating Lambda Warm Container\n")
    print("=" * 60)

    # Simulate multiple invocations (same container)
    test_events = [
        {"text": "What is machine learning?"},
        {"text": "How do neural networks work?"},
        {"text": "What is machine learning?"},  # Cache hit
        {"text": "Explain deep learning"},
        {"text": "How do neural networks work?"},  # Cache hit
        {"text": "What is machine learning?"},  # Cache hit
    ]

    print("\nInvocations on warm container:")
    print("-" * 60)

    for i, event in enumerate(test_events, 1):
        print(f"\n[Invocation {i}] {event['text'][:40]}...")

        result = lambda_handler(event, None)
        body = result["body"]

        status = "HIT ✓" if body["cache_hit"] else "MISS ✗"
        print(f"  Status: {status}")
        print(f"  Memory: {body['memory_mb']:.2f} MB")
        print(f"  Hot cache: {body['cached_vectors']}/{body['total_records']} vectors")

    print("\n" + "=" * 60)
    print("\nSingleton Pattern Benefits:")
    print("  • First call:  10ms cold start")
    print("  • Next calls:  0ms overhead (reuses instance)")
    print("  • Memory:      ~0.6 MB for 100 cached vectors")
    print("  • Lifetime:    ~15 minutes (Lambda warm container)")
    print("  • Savings:     Skip 50+ embedding API calls per container")

    print("\nMemory Efficiency:")
    print("  • Index only:  160 KB for 10K vectors")
    print("  • LRU cache:   0.6 MB for 100 hot vectors")
    print("  • Total RAM:   ~1 MB (fits Lambda 128MB tier)")

    # Show memory stats
    cache = EmbedCache.for_serverless(name="embeddings", dimension=1536)
    print(f"\nActual memory usage: {cache.get_memory_usage() / 1024 / 1024:.2f} MB")
    print(f"Cached vectors: {cache.get_cached_count()}")
    print(f"Index size: {cache.get_index_size() / 1024:.2f} KB")


if __name__ == "__main__":
    main()
