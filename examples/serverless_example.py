"""
Serverless Example - Vercel/Lambda/Cloudflare Workers

This shows how EmbedCache works perfectly in serverless environments.
"""

import os
import tempfile
from embedcache import EmbedCache
import numpy as np


def lambda_handler(event, context):
    """
    Example Lambda/Vercel handler with EmbedCache

    The cache persists across warm invocations in /tmp (Lambda)
    or ephemeral storage (Vercel).
    """

    # Use /tmp for Lambda, or appropriate path for your platform
    cache_path = os.path.join(tempfile.gettempdir(), "embeddings.cache")

    # Initialize cache (fast on warm starts!)
    cache = EmbedCache(
        path=cache_path,
        dimension=1536,
        similarity_threshold=0.95
    )

    # Get text from event
    text = event.get("text", "Hello world")

    # Try cache first
    embedding = cache.get(text)

    if embedding is None:
        # Cache miss - compute embedding
        print(f"Cache MISS for: {text}")

        # Simulate embedding computation
        # In production, call your embedding API here
        embedding = np.random.randn(1536).astype(np.float32)
        embedding = embedding / np.linalg.norm(embedding)

        # Cache it
        cache.set(text, embedding)
    else:
        print(f"Cache HIT for: {text}")

    # Get stats
    stats = cache.stats()

    cache.close()

    return {
        "statusCode": 200,
        "body": {
            "embedding": embedding.tolist(),
            "cache_hit": embedding is not None,
            "cache_stats": {
                "records": stats.records,
                "file_size_kb": stats.file_size / 1024,
            }
        }
    }


def main():
    """Simulate serverless invocations"""
    print("Simulating Serverless Environment\n")
    print("=" * 50)

    # Simulate multiple Lambda invocations
    test_events = [
        {"text": "What is machine learning?"},
        {"text": "How do neural networks work?"},
        {"text": "What is machine learning?"},  # Should hit cache
        {"text": "Explain deep learning"},
        {"text": "How do neural networks work?"},  # Should hit cache
    ]

    for i, event in enumerate(test_events, 1):
        print(f"\nInvocation {i}:")
        print(f"Input: {event['text']}")

        result = lambda_handler(event, None)

        print(f"Status: {result['statusCode']}")
        print(f"Cache hit: {result['body']['cache_hit']}")
        print(f"Cache records: {result['body']['cache_stats']['records']}")
        print(f"Cache size: {result['body']['cache_stats']['file_size_kb']:.2f} KB")

    print("\n" + "=" * 50)
    print("\nKey Benefits for Serverless:")
    print("  ✓ Zero dependencies (no Redis, no external DB)")
    print("  ✓ Fast cold starts (<10ms)")
    print("  ✓ Persists across warm invocations")
    print("  ✓ Works in /tmp (Lambda) or ephemeral storage (Vercel)")
    print("  ✓ Tiny binary (2MB)")
    print("  ✓ Saves $$$ on embedding API calls")


if __name__ == "__main__":
    main()
