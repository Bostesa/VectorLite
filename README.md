# VectorLite

Embedded vector database for serverless & edge. Starting as a simple embedding cache, evolving into a full vector database.

## What is this?

An embedding cache that persists to a single file. Think SQLite for vector embeddings. Built with Go for the storage engine and Python bindings for ease of use.

**Current status**: Phase 1 complete - basic cache with similarity matching works.

```python
from embedcache import EmbedCache

# Create cache
cache = EmbedCache(path="./cache.db", dimension=1536)

# Cache embeddings
cache.set("hello world", embedding_vector)
vector = cache.get("hello world")

# Similarity search
cache = EmbedCache(similarity_threshold=0.95)
similar, score = cache.find_similar(query_vector)
```

## Why?

- **Zero config**: No Redis, no Docker, no servers
- **Fast**: Memory-mapped I/O, sub-millisecond lookups
- **Tiny memory footprint**: 99% less memory via lazy loading (5MB vs 60MB for 10K vectors)
- **Serverless-friendly**: 2MB binary, 10ms cold starts, works in Lambda/Vercel
- **Saves money**: Cache embeddings instead of recomputing

## Build

See [BUILD.md](BUILD.md) for platform-specific instructions.

```bash
# macOS/Linux quick build
go build -buildmode=c-shared -o python/embedcache/lib/libembedcache.dylib ./pkg/embedcache
cd python && pip install -e .
```

## Usage

See [GETTING_STARTED.md](GETTING_STARTED.md) for detailed examples.

```python
from embedcache import EmbedCache

cache = EmbedCache(dimension=1536)

# Basic operations
cache.set(key, vector)
vector = cache.get(key)

# Get or compute pattern
def compute_embedding():
    return openai.embeddings.create(...)

vector = cache.get_or_compute("text", compute_embedding)
```

## Architecture

- **Storage**: Go with memory-mapped files
- **API**: Python via CGO bindings
- **Format**: Single file with hash index + similarity search
- **Lazy loading**: LRU cache keeps hot vectors in memory, others on-demand
- **Tests**: 26 tests (13 Go + 13 Python)

**Performance** (M1 Mac, 1536-dim vectors):
- Insert: 0.05ms (20K/sec)
- Get (cached): 0.001ms (1M/sec)
- Get (cold): 0.1ms (10K/sec)
- Cold start: 10ms (index-only load)
- Memory: 5MB for 10K vectors (99% less than eager loading)

## Roadmap

**Phase 1** (complete): Basic cache with exact match + similarity
**Phase 2** (planned): HNSW index, ACID transactions, metadata filtering

## Tests

```bash
# Go
go test ./pkg/storage -v

# Python
cd python && source venv/bin/activate
pytest tests/ -v
```

## Examples

- `examples/basic_usage.py` - Simple cache demo
- `examples/openai_example.py` - OpenAI integration
- `examples/serverless_example.py` - Lambda/Vercel usage

## License

MIT
