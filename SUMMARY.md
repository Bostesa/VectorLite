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
Storage: Single .edb file
├── Header (4KB): metadata
├── Records: hash → vector mappings
└── Index: in-memory hash table

Operations:
- Insert: Hash text, append to file, update index
- Get: Hash text, lookup in index, read from mmap
- Similarity: Linear scan, cosine distance
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
- [x] Python bindings
- [x] Tests (21 passing)

### Phase 2 (Planned)
- [ ] HNSW index (scale to millions)
- [ ] ACID transactions (WAL)
- [ ] Metadata filtering
- [ ] Compression (PQ)

## Performance

Tested on M1 Mac with 1536-dim vectors:
- Insert: 0.05ms
- Get: 0.1ms
- Similarity (1K items): 2ms

Storage: ~6KB per vector

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
**Linear similarity**: Simple, fast enough for Phase 1

## Status

Phase 1 complete. Works for basic use cases (< 10K vectors). Ready to use, ready to extend.
