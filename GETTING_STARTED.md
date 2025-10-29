# Getting Started

Quick guide to using EmbedCache.

## Install

```bash
git clone https://github.com/Bostesa/VectorLite.git
cd vectorlite

# Build (see BUILD.md for other platforms)
go build -buildmode=c-shared -o python/embedcache/lib/libembedcache.dylib ./pkg/embedcache

# Install
cd python && pip install -e .
```

## Basic Usage

```python
from embedcache import EmbedCache
import numpy as np

# Create cache
cache = EmbedCache(path="./cache.db", dimension=1536)

# Store
cache.set("hello", vector)

# Retrieve
vector = cache.get("hello")

# Close
cache.close()
```

## With OpenAI

```python
from openai import OpenAI
from embedcache import EmbedCache

client = OpenAI()
cache = EmbedCache(dimension=1536)

def get_embedding(text):
    # Try cache first
    cached = cache.get(text)
    if cached is not None:
        return cached

    # Compute and cache
    response = client.embeddings.create(input=text, model="text-embedding-ada-002")
    vector = response.data[0].embedding
    cache.set(text, vector)
    return vector

embedding = get_embedding("What is machine learning?")
```

## Similarity Matching

Enable to catch similar queries:

```python
cache = EmbedCache(
    dimension=1536,
    similarity_threshold=0.95  # 95% similarity
)

# Store
cache.set("hello world", vector1)

# Find similar
result = cache.find_similar(query_vector, threshold=0.95)
if result:
    cached_vector, score = result
    print(f"Found with {score:.2f} similarity")
```

## Serverless (Lambda/Vercel)

```python
import os
from embedcache import EmbedCache

def handler(event, context):
    # Use ephemeral storage
    cache = EmbedCache(path="/tmp/cache.db", dimension=1536)

    text = event["text"]

    # Check cache
    embedding = cache.get(text)
    if embedding is None:
        embedding = compute_embedding(text)
        cache.set(text, embedding)

    cache.close()
    return {"embedding": embedding.tolist()}
```

## API Reference

### EmbedCache

```python
EmbedCache(
    path: str = "./embeddings.cache",
    dimension: int = 1536,
    similarity_threshold: float = None
)
```

### Methods

**`set(key: str, vector: array) -> None`**

Store an embedding.

**`get(key: str) -> array | None`**

Retrieve an embedding. Returns None if not found.

**`find_similar(vector: array, threshold: float) -> (array, float) | None`**

Find most similar cached embedding. Returns (vector, score) or None.

**`get_or_compute(key: str, compute_fn: callable) -> array`**

Get from cache or compute if not found.

**`stats() -> CacheStats`**

Get cache statistics.

**`close() -> None`**

Close cache and flush to disk.

### Context Manager

```python
with EmbedCache(path="./cache.db") as cache:
    cache.set("key", vector)
    result = cache.get("key")
```

## Common Patterns

### Batch Processing

```python
for text, vector in zip(texts, vectors):
    cache.set(text, vector)
```

### Error Handling

```python
try:
    cache = EmbedCache(path="./cache.db", dimension=1536)
    # use cache
except Exception as e:
    print(f"Error: {e}")
finally:
    cache.close()
```

### Dimension Validation

```python
# All vectors must have same dimension
cache = EmbedCache(dimension=384)  # sentence-transformers
cache.set("text1", vector_384)     # OK
cache.set("text2", vector_1536)    # Error: dimension mismatch
```

## Performance Tips

1. **Reuse cache instance** - Don't create new cache for each operation
2. **Use similarity threshold wisely** - 0.95+ is safe, lower risks false matches
3. **Close properly** - Always close or use context manager
4. **Batch when possible** - Multiple sets are fine, cache handles buffering

## Troubleshooting

**"Could not find shared library"**

Build the Go library first (see BUILD.md).

**"Dimension mismatch"**

All vectors must match the dimension specified when creating the cache.

**"Failed to open cache"**

Check file permissions and path. Create parent directories if needed.

## Examples

See `examples/` directory:
- `basic_usage.py` - Simple demo
- `openai_example.py` - OpenAI integration
- `serverless_example.py` - Lambda/Vercel usage
