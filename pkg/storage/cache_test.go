package storage

import (
	"os"
	"testing"
)

func TestLRUCache_Basic(t *testing.T) {
	cache := NewLRUCache(3)

	// Add items
	cache.Put(1, []float32{1.0, 2.0})
	cache.Put(2, []float32{3.0, 4.0})
	cache.Put(3, []float32{5.0, 6.0})

	if cache.Len() != 3 {
		t.Errorf("Expected cache length 3, got %d", cache.Len())
	}

	// Get items
	if vec, ok := cache.Get(1); !ok || vec[0] != 1.0 {
		t.Error("Failed to get item 1")
	}
}

func TestLRUCache_Eviction(t *testing.T) {
	cache := NewLRUCache(2)

	cache.Put(1, []float32{1.0})
	cache.Put(2, []float32{2.0})
	cache.Put(3, []float32{3.0}) // Should evict 1

	if cache.Len() != 2 {
		t.Errorf("Expected cache length 2, got %d", cache.Len())
	}

	// Item 1 should be evicted
	if _, ok := cache.Get(1); ok {
		t.Error("Item 1 should have been evicted")
	}

	// Items 2 and 3 should still be there
	if _, ok := cache.Get(2); !ok {
		t.Error("Item 2 should still be in cache")
	}
	if _, ok := cache.Get(3); !ok {
		t.Error("Item 3 should still be in cache")
	}
}

func TestLRUCache_LRUOrder(t *testing.T) {
	cache := NewLRUCache(2)

	cache.Put(1, []float32{1.0})
	cache.Put(2, []float32{2.0})

	// Access item 1 to make it recently used
	cache.Get(1)

	// Add item 3 - should evict 2 (least recently used)
	cache.Put(3, []float32{3.0})

	// Item 2 should be evicted
	if _, ok := cache.Get(2); ok {
		t.Error("Item 2 should have been evicted")
	}

	// Items 1 and 3 should still be there
	if _, ok := cache.Get(1); !ok {
		t.Error("Item 1 should still be in cache")
	}
	if _, ok := cache.Get(3); !ok {
		t.Error("Item 3 should still be in cache")
	}
}

func TestDB_CacheHit(t *testing.T) {
	tmpfile := "test_cache.edb"
	defer os.Remove(tmpfile)

	db, err := Open(tmpfile, 3)
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Insert a vector
	text := "test"
	vector := []float32{0.1, 0.2, 0.3}
	db.Insert(text, vector)

	// First get - cache miss
	stats1 := db.Stats()
	cacheSize1 := stats1["cache_size"].(int)

	result1, _ := db.Get(text)
	if result1 == nil {
		t.Fatal("Failed to get vector")
	}

	// Check cache was populated
	stats2 := db.Stats()
	cacheSize2 := stats2["cache_size"].(int)

	if cacheSize2 <= cacheSize1 {
		t.Error("Cache should have been populated after first get")
	}

	// Second get - should hit cache
	result2, _ := db.Get(text)
	if result2 == nil {
		t.Fatal("Failed to get vector from cache")
	}

	// Verify results are same
	for i := range result1 {
		if result1[i] != result2[i] {
			t.Errorf("Cache returned different result")
		}
	}
}

func TestDB_OpenWithOptions(t *testing.T) {
	tmpfile := "test_options.edb"
	defer os.Remove(tmpfile)

	// Create with custom cache size
	opts := OpenOptions{
		LazyLoad:  true,
		CacheSize: 50,
	}

	db, err := OpenWithOptions(tmpfile, 3, opts)
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	stats := db.Stats()
	if stats["cache_capacity"].(int) != 50 {
		t.Errorf("Expected cache capacity 50, got %d", stats["cache_capacity"].(int))
	}
}

func BenchmarkDB_GetWithCache(b *testing.B) {
	tmpfile := "bench_cache.edb"
	defer os.Remove(tmpfile)

	db, _ := Open(tmpfile, 1536)
	defer db.Close()

	// Insert 1000 vectors
	vector := make([]float32, 1536)
	for i := 0; i < 1000; i++ {
		for j := range vector {
			vector[j] = float32(i+j) / 1536.0
		}
		db.Insert(string(rune(i)), vector)
	}

	// Access first 100 repeatedly (should all be cached)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		text := string(rune(i % 100))
		db.Get(text)
	}
}
