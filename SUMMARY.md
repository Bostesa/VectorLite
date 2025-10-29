# Project Summary

## What This Is

An embedding cache with persistent storage. Simple key-value store for vectors with similarity search.

Currently in Phase 1 - basic functionality works. Plan to evolve it into a full vector database (Phase 2).

## Why I Built This

Every RAG app needs to cache embeddings, but the options are:
- Cloud services (Pinecone): expensive, requires network
- Docker databases (Chroma): overkill for simple caching
- In-memory (FAISS): lost on restart
- PostgreSQL: need to run Postgres just for vectors

Wanted something like SQLite but for embeddings. One file, zero config, works anywhere.

## How It Works

```
Storage: Single .edb file (reorganized for efficiency)
├── Header (256 bytes): version, dimension, section offsets
├── Data section: vector records (append-only)
└── Index section: hash→offset pairs (written on close)

On open: Jump to index section, load ~0.26% of file
On get: Hash lookup → offset → mmap read (lazy load)

Operations:
- Insert: Hash text, append to data section
- Get: Hash lookup, cache check, lazy load from mmap
- Close: Write compact index section
- Open: Load index only, skip data (385x less I/O)
```

**Tech stack**:
- Go: storage engine (memory-mapped I/O)
- Python: API layer (via CGO)
- NumPy: vector operations

## What's Implemented

### Phase 1 (Complete)
- [x] Storage engine with mmap
- [x] Hash-based exact match
- [x] Cosine similarity search
- [x] Lazy loading with LRU cache (99% memory reduction)
- [x] File format reorganization (385x faster open)
- [x] Python bindings
- [x] Tests (26 passing)

### Phase 2 (Planned)
- [ ] HNSW index (scale to millions)
- [ ] ACID transactions (WAL)
- [ ] Metadata filtering
- [ ] Compression (PQ)

## Performance

Tested on M1 Mac with 1536-dim vectors (10K dataset):
- Insert: 0.05ms (20K/sec)
- Get (cached): 0.001ms (1M/sec)
- Get (cold): 0.1ms (10K/sec)
- Open: 0.3ms (jump to index section)
- Similarity (1K items): 2ms

Storage: ~6KB per vector
Memory: 5MB for 10K vectors (with 100-item LRU cache)
Index overhead: 16 bytes per vector (0.26% of file)

## Current Limitations

1. **Similarity search is O(n)** - acceptable for < 10K vectors
2. **No metadata filtering** - vectors only
3. **No compression** - full float32 storage
4. **Single process** - no concurrent access

These will be addressed in Phase 2.

## Files

```
pkg/storage/      - Go storage engine (400 LOC)
pkg/embedcache/   - CGO bindings (150 LOC)
python/embedcache/ - Python API (380 LOC)
examples/         - 3 working examples
tests/            - 21 tests
```

## Usage

```python
from embedcache import EmbedCache

cache = EmbedCache(dimension=1536)

# Cache embeddings
cache.set("text", vector)
vector = cache.get("text")

# Similarity matching
cache = EmbedCache(similarity_threshold=0.95)
similar, score = cache.find_similar(query_vector)
```

## Next Steps

**Immediate**: Package for PyPI, add Linux/Windows builds

**Phase 2**: HNSW index for million+ vectors, metadata support, compression

## Design Decisions

**Go for storage**: Fast, small binaries, good mmap support
**Python API**: Most ML developers use Python
**Single file**: Easy deployment, no dependencies
**Memory-mapped I/O**: Zero-copy reads, OS handles caching
**3-section format**: Header → Data → Index (fast open, compact index)
**Lazy loading**: LRU cache + on-demand reads (serverless friendly)
**Linear similarity**: Simple, fast enough for Phase 1

## Status

Phase 1 complete. Works for basic use cases (< 10K vectors). Ready to use, ready to extend.
