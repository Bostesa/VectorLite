"""Tests for EmbedCache"""

import os
import tempfile
import pytest
import numpy as np

from embedcache import EmbedCache


@pytest.fixture
def temp_cache():
    """Create a temporary cache for testing"""
    fd, path = tempfile.mkstemp(suffix=".cache")
    os.close(fd)
    os.unlink(path)  # Remove the file, let EmbedCache create it

    cache = EmbedCache(path=path, dimension=3)
    yield cache

    cache.close()
    if os.path.exists(path):
        os.unlink(path)


def test_create_cache(temp_cache):
    """Test cache creation"""
    assert temp_cache is not None
    assert temp_cache.dimension == 3


def test_set_and_get(temp_cache):
    """Test basic set and get operations"""
    key = "hello world"
    vector = [0.1, 0.2, 0.3]

    # Set
    temp_cache.set(key, vector)

    # Get
    result = temp_cache.get(key)
    assert result is not None
    np.testing.assert_array_almost_equal(result, vector)


def test_get_nonexistent(temp_cache):
    """Test getting a non-existent key"""
    result = temp_cache.get("nonexistent")
    assert result is None


def test_numpy_arrays(temp_cache):
    """Test with numpy arrays"""
    key = "numpy test"
    vector = np.array([0.5, 0.6, 0.7], dtype=np.float32)

    temp_cache.set(key, vector)
    result = temp_cache.get(key)

    assert result is not None
    np.testing.assert_array_almost_equal(result, vector)


def test_multiple_keys(temp_cache):
    """Test multiple keys"""
    data = {
        "first": [0.1, 0.2, 0.3],
        "second": [0.4, 0.5, 0.6],
        "third": [0.7, 0.8, 0.9],
    }

    # Insert all
    for key, vector in data.items():
        temp_cache.set(key, vector)

    # Verify all
    for key, expected in data.items():
        result = temp_cache.get(key)
        assert result is not None
        np.testing.assert_array_almost_equal(result, expected)


def test_dimension_mismatch(temp_cache):
    """Test dimension mismatch error"""
    with pytest.raises(ValueError, match="dimension mismatch"):
        temp_cache.set("test", [0.1, 0.2])  # Wrong dimension


def test_find_similar():
    """Test similarity matching"""
    fd, path = tempfile.mkstemp(suffix=".cache")
    os.close(fd)
    os.unlink(path)

    cache = EmbedCache(path=path, dimension=3, similarity_threshold=0.9)

    # Insert some vectors
    cache.set("hello", [1.0, 0.0, 0.0])
    cache.set("world", [0.0, 1.0, 0.0])

    # Find similar to "hello"
    query = [0.95, 0.1, 0.05]
    result, score = cache.find_similar(query)

    assert result is not None
    assert score >= 0.9

    cache.close()
    os.unlink(path)


def test_stats(temp_cache):
    """Test statistics"""
    # Insert some data
    temp_cache.set("key1", [0.1, 0.2, 0.3])
    temp_cache.set("key2", [0.4, 0.5, 0.6])

    stats = temp_cache.stats()
    assert stats.records == 2
    assert stats.dimension == 3
    assert stats.file_size > 0


def test_persistence():
    """Test that data persists after close and reopen"""
    fd, path = tempfile.mkstemp(suffix=".cache")
    os.close(fd)
    os.unlink(path)

    key = "persistent"
    vector = [0.1, 0.2, 0.3]

    # Create, insert, and close
    cache1 = EmbedCache(path=path, dimension=3)
    cache1.set(key, vector)
    cache1.close()

    # Reopen and verify
    cache2 = EmbedCache(path=path, dimension=3)
    result = cache2.get(key)

    assert result is not None
    np.testing.assert_array_almost_equal(result, vector)

    cache2.close()
    os.unlink(path)


def test_get_or_compute(temp_cache):
    """Test get_or_compute functionality"""
    key = "compute_test"
    vector = [0.9, 0.8, 0.7]

    call_count = 0

    def compute():
        nonlocal call_count
        call_count += 1
        return vector

    # First call should compute
    result1 = temp_cache.get_or_compute(key, compute, use_similarity=False)
    assert call_count == 1
    np.testing.assert_array_almost_equal(result1, vector)

    # Second call should use cache
    result2 = temp_cache.get_or_compute(key, compute, use_similarity=False)
    assert call_count == 1  # Should not have called compute again
    np.testing.assert_array_almost_equal(result2, vector)


def test_context_manager():
    """Test using cache as context manager"""
    fd, path = tempfile.mkstemp(suffix=".cache")
    os.close(fd)
    os.unlink(path)

    with EmbedCache(path=path, dimension=3) as cache:
        cache.set("test", [0.1, 0.2, 0.3])
        result = cache.get("test")
        assert result is not None

    # Cache should be closed now
    os.unlink(path)


def test_repr(temp_cache):
    """Test string representation"""
    repr_str = repr(temp_cache)
    assert "EmbedCache" in repr_str
    assert "dimension=3" in repr_str


def test_large_dimension():
    """Test with realistic embedding dimension (OpenAI ada-002)"""
    fd, path = tempfile.mkstemp(suffix=".cache")
    os.close(fd)
    os.unlink(path)

    cache = EmbedCache(path=path, dimension=1536)

    # Create a realistic vector
    vector = np.random.randn(1536).astype(np.float32)
    vector = vector / np.linalg.norm(vector)  # Normalize

    cache.set("test_embedding", vector)
    result = cache.get("test_embedding")

    assert result is not None
    assert len(result) == 1536
    np.testing.assert_array_almost_equal(result, vector, decimal=6)

    cache.close()
    os.unlink(path)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
