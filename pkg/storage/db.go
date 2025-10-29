package storage

import (
	"encoding/binary"
	"errors"
	"fmt"
	"math"
	"os"
	"sync"
	"syscall"
)

var (
	ErrNotFound          = errors.New("embedding not found")
	ErrInvalidFormat     = errors.New("invalid file format")
	ErrDimensionMismatch = errors.New("vector dimension mismatch")
)

type DB struct {
	path      string
	file      *os.File
	mmap      []byte
	header    *Header
	index     map[uint64]int64 // hash -> offset (lightweight)
	cache     *LRUCache         // hot vectors only
	mu        sync.RWMutex
	dimension uint32
}

type OpenOptions struct {
	LazyLoad  bool // Only load index, not vectors
	CacheSize int  // LRU cache size (default 100)
}

func Open(path string, dimension uint32) (*DB, error) {
	return OpenWithOptions(path, dimension, OpenOptions{
		LazyLoad:  true,
		CacheSize: 100,
	})
}

func OpenWithOptions(path string, dimension uint32, opts OpenOptions) (*DB, error) {
	if opts.CacheSize == 0 {
		opts.CacheSize = 100
	}

	db := &DB{
		path:      path,
		index:     make(map[uint64]int64),
		cache:     NewLRUCache(opts.CacheSize),
		dimension: dimension,
	}

	_, err := os.Stat(path)
	if os.IsNotExist(err) {
		return db.create()
	} else if err != nil {
		return nil, err
	}

	return db.open()
}

func (db *DB) create() (*DB, error) {
	f, err := os.Create(db.path)
	if err != nil {
		return nil, err
	}
	db.file = f

	header := NewHeader(db.dimension)
	headerBytes := header.Encode()
	if _, err := f.Write(headerBytes); err != nil {
		f.Close()
		return nil, err
	}

	db.header = header
	return db, nil
}

func (db *DB) open() (*DB, error) {
	f, err := os.OpenFile(db.path, os.O_RDWR, 0644)
	if err != nil {
		return nil, err
	}
	db.file = f

	headerBytes := make([]byte, HeaderSize)
	if _, err := f.ReadAt(headerBytes, 0); err != nil {
		f.Close()
		return nil, err
	}

	header := DecodeHeader(headerBytes)
	if string(header.Magic[:]) != MagicBytes {
		f.Close()
		return nil, ErrInvalidFormat
	}

	if header.Dimension != db.dimension {
		f.Close()
		return nil, fmt.Errorf("%w: expected %d, got %d",
			ErrDimensionMismatch, db.dimension, header.Dimension)
	}

	db.header = header

	stat, err := f.Stat()
	if err != nil {
		f.Close()
		return nil, err
	}

	if stat.Size() > HeaderSize {
		db.mmap, err = syscall.Mmap(
			int(f.Fd()),
			0,
			int(stat.Size()),
			syscall.PROT_READ|syscall.PROT_WRITE,
			syscall.MAP_SHARED,
		)
		if err != nil {
			f.Close()
			return nil, err
		}

		if err := db.buildIndex(); err != nil {
			db.Close()
			return nil, err
		}
	}

	return db, nil
}

func (db *DB) buildIndex() error {
	offset := int64(HeaderSize)
	fileSize := int64(len(db.mmap))

	for offset < fileSize {
		if offset+RecordMetaSize > fileSize {
			break
		}

		hash := binary.LittleEndian.Uint64(db.mmap[offset : offset+8])
		dim := binary.LittleEndian.Uint32(db.mmap[offset+8 : offset+12])

		db.index[hash] = offset

		recordSize := RecordMetaSize + int64(dim)*4
		offset += recordSize
	}

	return nil
}

func (db *DB) Insert(text string, vector []float32) error {
	if len(vector) != int(db.dimension) {
		return fmt.Errorf("%w: expected %d, got %d",
			ErrDimensionMismatch, db.dimension, len(vector))
	}

	db.mu.Lock()
	defer db.mu.Unlock()

	hash := HashText(text)

	if _, exists := db.index[hash]; exists {
		return nil
	}

	recordSize := RecordMetaSize + len(vector)*4
	buf := make([]byte, recordSize)

	binary.LittleEndian.PutUint64(buf[0:8], hash)
	binary.LittleEndian.PutUint32(buf[8:12], uint32(len(vector)))

	for i, v := range vector {
		bits := math.Float32bits(v)
		binary.LittleEndian.PutUint32(buf[RecordMetaSize+i*4:], bits)
	}

	stat, err := db.file.Stat()
	if err != nil {
		return err
	}

	offset := stat.Size()

	if _, err := db.file.WriteAt(buf, offset); err != nil {
		return err
	}

	db.index[hash] = offset
	db.header.RecordCount++

	if err := db.remap(); err != nil {
		return err
	}

	return nil
}

func (db *DB) Get(text string) ([]float32, error) {
	hash := HashText(text)

	// Check LRU cache first (no lock needed)
	if cached, ok := db.cache.Get(hash); ok {
		return cached, nil
	}

	// Cache miss - read from mmap
	db.mu.RLock()
	offset, exists := db.index[hash]
	db.mu.RUnlock()

	if !exists {
		return nil, ErrNotFound
	}

	// Read vector from mmap (lock held during read)
	db.mu.RLock()
	vector, err := db.readVector(offset)
	db.mu.RUnlock()

	if err != nil {
		return nil, err
	}

	// Add to cache for next time
	db.cache.Put(hash, vector)

	return vector, nil
}

func (db *DB) readVector(offset int64) ([]float32, error) {
	if db.mmap == nil {
		return nil, ErrNotFound
	}

	dim := binary.LittleEndian.Uint32(db.mmap[offset+8 : offset+12])

	vector := make([]float32, dim)
	vectorOffset := offset + RecordMetaSize

	for i := range vector {
		bits := binary.LittleEndian.Uint32(db.mmap[vectorOffset+int64(i)*4:])
		vector[i] = math.Float32frombits(bits)
	}

	return vector, nil
}

func (db *DB) FindSimilar(vector []float32, threshold float32) ([]float32, float32, error) {
	if len(vector) != int(db.dimension) {
		return nil, 0, ErrDimensionMismatch
	}

	db.mu.RLock()
	offsets := make([]int64, 0, len(db.index))
	for _, offset := range db.index {
		offsets = append(offsets, offset)
	}
	db.mu.RUnlock()

	var bestVector []float32
	var bestScore float32 = -1

	// O(n) linear scan - will add HNSW in Phase 2
	for _, offset := range offsets {
		db.mu.RLock()
		cached, err := db.readVector(offset)
		db.mu.RUnlock()

		if err != nil {
			continue
		}

		score := cosineSimilarity(vector, cached)
		if score > bestScore {
			bestScore = score
			bestVector = cached
		}
	}

	if bestScore >= threshold {
		return bestVector, bestScore, nil
	}

	return nil, 0, ErrNotFound
}

func cosineSimilarity(a, b []float32) float32 {
	if len(a) != len(b) {
		return -1
	}

	var dotProduct, normA, normB float32
	for i := range a {
		dotProduct += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return dotProduct / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

func (db *DB) remap() error {
	if db.mmap != nil {
		if err := syscall.Munmap(db.mmap); err != nil {
			return err
		}
	}

	stat, err := db.file.Stat()
	if err != nil {
		return err
	}

	if stat.Size() <= HeaderSize {
		db.mmap = nil
		return nil
	}

	db.mmap, err = syscall.Mmap(
		int(db.file.Fd()),
		0,
		int(stat.Size()),
		syscall.PROT_READ|syscall.PROT_WRITE,
		syscall.MAP_SHARED,
	)

	return err
}

func (db *DB) Stats() map[string]interface{} {
	db.mu.RLock()
	defer db.mu.RUnlock()

	stat, _ := db.file.Stat()

	return map[string]interface{}{
		"records":     len(db.index),
		"dimension":   db.dimension,
		"file_size":   stat.Size(),
		"index_size":  len(db.index),
		"cache_size":  db.cache.Len(),
		"cache_capacity": db.cache.capacity,
	}
}

func (db *DB) Close() error {
	db.mu.Lock()
	defer db.mu.Unlock()

	headerBytes := db.header.Encode()
	if _, err := db.file.WriteAt(headerBytes, 0); err != nil {
		return err
	}

	if db.mmap != nil {
		if err := syscall.Munmap(db.mmap); err != nil {
			return err
		}
	}

	return db.file.Close()
}
