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

# Create cache (default: 100 vectors cached in memory)
cache = EmbedCache(path="./cache.db", dimension=1536)

# Create with custom cache size
cache = EmbedCache(path="./cache.db", dimension=1536, cache_size=500)

# Store
cache.set("hello", vector)

# Retrieve
vector = cache.get("hello")

# Close
cache.close()
```

## Serverless Usage (Recommended)

```python
from embedcache import EmbedCache

def lambda_handler(event, context):
    # Singleton pattern - reuses cache across warm invocations
    cache = EmbedCache.for_serverless(
        name="embeddings",
        dimension=1536,
        cache_size=100  # Adjust based on memory budget
    )

    # First invocation: 10ms cold start
    # Next 50+ invocations: 0ms (reuses instance)

    text = event["text"]
    embedding = cache.get(text)

    if embedding is None:
        embedding = compute_embedding(text)  # Your embedding API
        cache.set(text, embedding)

    # Don't close! Instance persists for next invocation
    return {"embedding": embedding.tolist()}
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

## Serverless Optimization

EmbedCache uses singleton pattern to reuse cache across Lambda warm invocations:

```python
from embedcache import EmbedCache

def handler(event, context):
    # Singleton - persists across warm invocations
    cache = EmbedCache.for_serverless(name="embeddings", dimension=1536)

    # First call: 10ms to open
    # Next 50+ calls: 0ms (reuses existing instance)

    text = event["text"]
    embedding = cache.get(text)

    if embedding is None:
        embedding = compute_embedding(text)
        cache.set(text, embedding)

    # Don't close! Keep alive for next invocation
    return {"embedding": embedding.tolist()}
```

**Benefits:**
- First invocation: 10ms cold start
- Subsequent invocations: 0ms overhead
- Cache persists ~15 minutes (Lambda warm container)
- Saves 50+ embedding API calls per container
- ~1 MB memory (index + 100 cached vectors)

## API Reference

### EmbedCache

```python
EmbedCache(
    path: str = "./embeddings.cache",
    dimension: int = 1536,
    similarity_threshold: float = None,
    cache_size: int = 100  # LRU cache size
)
```

**Parameters:**
- `path`: Database file path
- `dimension`: Vector dimension (must match all vectors)
- `similarity_threshold`: Default similarity threshold for find_similar()
- `cache_size`: Number of vectors to keep in memory (default: 100)

### Methods

**`set(key: str, vector: array) -> None`**

Store an embedding.

**`get(key: str) -> array | None`**

Retrieve an embedding. Returns None if not found.

**`find_similar(vector: array, threshold: float) -> (array, float) | None`**

Find most similar cached embedding. Returns (vector, score) or None.

**`get_or_compute(key: str, compute_fn: callable) -> array`**

Get from cache or compute if not found.

**`get_memory_usage() -> int`**

Get actual RAM usage in bytes (index + cached vectors).

**`get_cached_count() -> int`**

Get number of vectors currently in hot cache (LRU).

**`get_index_size() -> int`**

Get size of index in memory (hash table).

**`stats() -> CacheStats`**

Get cache statistics including memory metrics.

**`close() -> None`**

Close cache and flush to disk.

### Class Methods

**`EmbedCache.for_serverless(name, dimension, cache_size) -> EmbedCache`**

Create or reuse cache instance optimized for serverless. Uses singleton pattern to persist across warm Lambda invocations.

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
5. **Tune cache_size** - Increase for hot data sets, decrease for memory-constrained environments

## Memory Optimization

EmbedCache uses lazy loading with an LRU cache:

```python
# Default: 100 vectors in memory (good for most cases)
cache = EmbedCache(dimension=1536)

# High traffic: keep more vectors hot
cache = EmbedCache(dimension=1536, cache_size=500)

# Memory-constrained (Lambda 128MB): keep minimal cache
cache = EmbedCache(dimension=1536, cache_size=10)
```

**Memory usage** (1536-dim vectors):
- 10 vectors in cache: 0.06 MB
- 100 vectors in cache: 0.6 MB
- 500 vectors in cache: 3 MB

Compare to loading all vectors:
- 10K vectors fully loaded: 60 MB
- 10K vectors lazy loaded: 5 MB (index + 100-item cache)

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
