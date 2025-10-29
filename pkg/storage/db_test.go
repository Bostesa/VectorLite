package storage

import (
	"errors"
	"os"
	"testing"
)

func TestDB_CreateAndOpen(t *testing.T) {
	tmpfile := "test_create.edb"
	defer os.Remove(tmpfile)

	// Create new database
	db, err := Open(tmpfile, 3)
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	if db.dimension != 3 {
		t.Errorf("Expected dimension 3, got %d", db.dimension)
	}

	if db.header.Version != CurrentVersion {
		t.Errorf("Expected version %d, got %d", CurrentVersion, db.header.Version)
	}
}

func TestDB_InsertAndGet(t *testing.T) {
	tmpfile := "test_insert.edb"
	defer os.Remove(tmpfile)

	db, err := Open(tmpfile, 3)
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Insert embedding
	text := "hello world"
	vector := []float32{0.1, 0.2, 0.3}

	if err := db.Insert(text, vector); err != nil {
		t.Fatalf("Failed to insert: %v", err)
	}

	// Retrieve embedding
	retrieved, err := db.Get(text)
	if err != nil {
		t.Fatalf("Failed to get: %v", err)
	}

	if len(retrieved) != len(vector) {
		t.Fatalf("Expected length %d, got %d", len(vector), len(retrieved))
	}

	for i := range vector {
		if retrieved[i] != vector[i] {
			t.Errorf("Vector mismatch at index %d: expected %f, got %f",
				i, vector[i], retrieved[i])
		}
	}
}

func TestDB_MultipleInserts(t *testing.T) {
	tmpfile := "test_multiple.edb"
	defer os.Remove(tmpfile)

	db, err := Open(tmpfile, 4)
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Insert multiple embeddings
	data := map[string][]float32{
		"first":  {0.1, 0.2, 0.3, 0.4},
		"second": {0.5, 0.6, 0.7, 0.8},
		"third":  {0.9, 1.0, 1.1, 1.2},
	}

	for text, vector := range data {
		if err := db.Insert(text, vector); err != nil {
			t.Fatalf("Failed to insert '%s': %v", text, err)
		}
	}

	// Verify all can be retrieved
	for text, expected := range data {
		retrieved, err := db.Get(text)
		if err != nil {
			t.Fatalf("Failed to get '%s': %v", text, err)
		}

		for i := range expected {
			if retrieved[i] != expected[i] {
				t.Errorf("Mismatch for '%s' at index %d: expected %f, got %f",
					text, i, expected[i], retrieved[i])
			}
		}
	}

	// Check stats
	stats := db.Stats()
	if stats["records"].(int) != 3 {
		t.Errorf("Expected 3 records, got %d", stats["records"].(int))
	}
}

func TestDB_NotFound(t *testing.T) {
	tmpfile := "test_notfound.edb"
	defer os.Remove(tmpfile)

	db, err := Open(tmpfile, 2)
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	_, err = db.Get("nonexistent")
	if err != ErrNotFound {
		t.Errorf("Expected ErrNotFound, got %v", err)
	}
}

func TestDB_DimensionMismatch(t *testing.T) {
	tmpfile := "test_dimension.edb"
	defer os.Remove(tmpfile)

	db, err := Open(tmpfile, 3)
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Try to insert wrong dimension
	err = db.Insert("test", []float32{0.1, 0.2})
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("Expected ErrDimensionMismatch, got %v", err)
	}
}

func TestDB_Persistence(t *testing.T) {
	tmpfile := "test_persist.edb"
	defer os.Remove(tmpfile)

	// Create and insert
	db1, err := Open(tmpfile, 3)
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}

	text := "persistent data"
	vector := []float32{1.1, 2.2, 3.3}
	if err := db1.Insert(text, vector); err != nil {
		t.Fatalf("Failed to insert: %v", err)
	}
	db1.Close()

	// Reopen and verify
	db2, err := Open(tmpfile, 3)
	if err != nil {
		t.Fatalf("Failed to reopen database: %v", err)
	}
	defer db2.Close()

	retrieved, err := db2.Get(text)
	if err != nil {
		t.Fatalf("Failed to get after reopen: %v", err)
	}

	for i := range vector {
		if retrieved[i] != vector[i] {
			t.Errorf("Mismatch after reopen at index %d: expected %f, got %f",
				i, vector[i], retrieved[i])
		}
	}
}

func TestCosineSimilarity(t *testing.T) {
	tests := []struct {
		name     string
		a        []float32
		b        []float32
		expected float32
		delta    float32
	}{
		{
			name:     "identical vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{1, 0, 0},
			expected: 1.0,
			delta:    0.0001,
		},
		{
			name:     "orthogonal vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{0, 1, 0},
			expected: 0.0,
			delta:    0.0001,
		},
		{
			name:     "opposite vectors",
			a:        []float32{1, 0, 0},
			b:        []float32{-1, 0, 0},
			expected: -1.0,
			delta:    0.0001,
		},
		{
			name:     "similar vectors",
			a:        []float32{1, 1, 0},
			b:        []float32{1, 0.9, 0.1},
			expected: 0.96,
			delta:    0.05,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := cosineSimilarity(tt.a, tt.b)
			diff := result - tt.expected
			if diff < 0 {
				diff = -diff
			}
			if diff > tt.delta {
				t.Errorf("Expected similarity %f, got %f (delta: %f)",
					tt.expected, result, diff)
			}
		})
	}
}

func TestDB_FindSimilar(t *testing.T) {
	tmpfile := "test_similar.edb"
	defer os.Remove(tmpfile)

	db, err := Open(tmpfile, 3)
	if err != nil {
		t.Fatalf("Failed to create database: %v", err)
	}
	defer db.Close()

	// Insert some embeddings
	db.Insert("hello world", []float32{1.0, 0.0, 0.0})
	db.Insert("hello there", []float32{0.9, 0.1, 0.0})
	db.Insert("goodbye", []float32{0.0, 1.0, 0.0})

	// Query with a vector similar to "hello world"
	query := []float32{0.95, 0.05, 0.0}
	result, score, err := db.FindSimilar(query, 0.85)

	if err != nil {
		t.Fatalf("Failed to find similar: %v", err)
	}

	if score < 0.85 {
		t.Errorf("Expected score >= 0.85, got %f", score)
	}

	// Result should be close to one of the hello vectors
	t.Logf("Found similar vector with score: %f", score)
	t.Logf("Result: %v", result)
}

func BenchmarkInsert(b *testing.B) {
	tmpfile := "bench_insert.edb"
	defer os.Remove(tmpfile)

	db, _ := Open(tmpfile, 1536) // OpenAI embedding size
	defer db.Close()

	vector := make([]float32, 1536)
	for i := range vector {
		vector[i] = float32(i) / 1536.0
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		text := string(rune(i))
		db.Insert(text, vector)
	}
}

func BenchmarkGet(b *testing.B) {
	tmpfile := "bench_get.edb"
	defer os.Remove(tmpfile)

	db, _ := Open(tmpfile, 1536)
	defer db.Close()

	vector := make([]float32, 1536)
	for i := range vector {
		vector[i] = float32(i) / 1536.0
	}

	// Insert 1000 vectors
	for i := 0; i < 1000; i++ {
		text := string(rune(i))
		db.Insert(text, vector)
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		text := string(rune(i % 1000))
		db.Get(text)
	}
}

func BenchmarkFindSimilar(b *testing.B) {
	tmpfile := "bench_similar.edb"
	defer os.Remove(tmpfile)

	db, _ := Open(tmpfile, 128)
	defer db.Close()

	// Insert 100 random vectors
	for i := 0; i < 100; i++ {
		vector := make([]float32, 128)
		for j := range vector {
			vector[j] = float32(i+j) / 128.0
		}
		db.Insert(string(rune(i)), vector)
	}

	query := make([]float32, 128)
	for i := range query {
		query[i] = 0.5
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		db.FindSimilar(query, 0.8)
	}
}
