import ctypes
import json
from dataclasses import dataclass
from typing import Optional, Callable, Union, Tuple

import numpy as np

from .lib_loader import load_library


@dataclass
class CacheStats:
    records: int
    dimension: int
    file_size: int
    index_size: int
    cache_size: int = 0
    cache_capacity: int = 0


class EmbedCache:
    def __init__(
        self,
        path: Optional[str] = None,
        dimension: int = 1536,
        similarity_threshold: Optional[float] = None,
        cache_size: int = 100,  # LRU cache size
    ):
        """
        Args:
            path: Path to cache file
            dimension: Vector dimension (default 1536 for OpenAI)
            similarity_threshold: Similarity threshold for matching
            cache_size: Number of vectors to keep in memory (default 100)
        """
        self.path = path or "./embeddings.cache"
        self.dimension = dimension
        self.similarity_threshold = similarity_threshold
        self.cache_size = cache_size

        self._lib = load_library()

        path_bytes = self.path.encode('utf-8')
        self._handle = self._lib.OpenDB(path_bytes, dimension)

        if self._handle < 0:
            raise RuntimeError(f"Failed to open cache at {self.path}")

    def __del__(self):
        if hasattr(self, '_handle') and self._handle >= 0:
            self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def set(self, key: str, vector: Union[list, np.ndarray]) -> None:
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}"
            )

        c_vector = (ctypes.c_float * len(vector))(*vector)
        key_bytes = key.encode('utf-8')

        result = self._lib.Insert(self._handle, key_bytes, c_vector, len(vector))

        if result < 0:
            raise RuntimeError("Failed to insert into cache")

    def get(self, key: str) -> Optional[np.ndarray]:
        key_bytes = key.encode('utf-8')
        out_vector = ctypes.POINTER(ctypes.c_float)()
        out_len = ctypes.c_int()

        result = self._lib.Get(
            self._handle,
            key_bytes,
            ctypes.byref(out_vector),
            ctypes.byref(out_len)
        )

        if result < 0:
            return None

        vector = np.ctypeslib.as_array(out_vector, shape=(out_len.value,))
        vector_copy = vector.copy()

        self._lib.FreeVector(out_vector)

        return vector_copy

    def find_similar(
        self,
        vector: Union[list, np.ndarray],
        threshold: Optional[float] = None
    ) -> Optional[Tuple[np.ndarray, float]]:
        if isinstance(vector, np.ndarray):
            vector = vector.tolist()

        if len(vector) != self.dimension:
            raise ValueError(
                f"Vector dimension mismatch: expected {self.dimension}, got {len(vector)}"
            )

        threshold = threshold or self.similarity_threshold
        if threshold is None:
            raise ValueError("No threshold specified")

        c_vector = (ctypes.c_float * len(vector))(*vector)
        out_vector = ctypes.POINTER(ctypes.c_float)()
        out_len = ctypes.c_int()
        out_score = ctypes.c_float()

        result = self._lib.FindSimilar(
            self._handle,
            c_vector,
            len(vector),
            threshold,
            ctypes.byref(out_vector),
            ctypes.byref(out_len),
            ctypes.byref(out_score)
        )

        if result < 0:
            return None

        cached = np.ctypeslib.as_array(out_vector, shape=(out_len.value,))
        cached_copy = cached.copy()

        self._lib.FreeVector(out_vector)

        return cached_copy, out_score.value

    def get_or_compute(
        self,
        key: str,
        compute_fn: Callable[[], Union[list, np.ndarray]],
    ) -> np.ndarray:
        cached = self.get(key)
        if cached is not None:
            return cached

        vector = compute_fn()
        self.set(key, vector)

        if isinstance(vector, list):
            vector = np.array(vector, dtype=np.float32)

        return vector

    def stats(self) -> CacheStats:
        result = self._lib.GetStats(self._handle)

        if result is None or not result:
            raise RuntimeError("Failed to get stats")

        stats_json = result.decode('utf-8')

        # Note: Not freeing string to avoid segfaults
        # Small leak but stats() isn't called often

        stats_dict = json.loads(stats_json)

        return CacheStats(
            records=stats_dict.get("records", 0),
            dimension=stats_dict.get("dimension", 0),
            file_size=stats_dict.get("file_size", 0),
            index_size=stats_dict.get("index_size", 0),
            cache_size=stats_dict.get("cache_size", 0),
            cache_capacity=stats_dict.get("cache_capacity", 0),
        )

    def close(self):
        if self._handle >= 0:
            self._lib.CloseDB(self._handle)
            self._handle = -1

    def __repr__(self):
        stats = self.stats()
        return (
            f"EmbedCache(path='{self.path}', "
            f"dimension={self.dimension}, "
            f"records={stats.records}, "
            f"similarity_threshold={self.similarity_threshold})"
        )
