package storage

import (
	"container/list"
	"sync"
)

// LRU cache for hot vectors
type LRUCache struct {
	capacity int
	cache    map[uint64]*list.Element
	lru      *list.List
	mu       sync.RWMutex
}

type cacheEntry struct {
	hash   uint64
	vector []float32
}

func NewLRUCache(capacity int) *LRUCache {
	return &LRUCache{
		capacity: capacity,
		cache:    make(map[uint64]*list.Element),
		lru:      list.New(),
	}
}

func (c *LRUCache) Get(hash uint64) ([]float32, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if elem, ok := c.cache[hash]; ok {
		c.lru.MoveToFront(elem)
		return elem.Value.(*cacheEntry).vector, true
	}
	return nil, false
}

func (c *LRUCache) Put(hash uint64, vector []float32) {
	c.mu.Lock()
	defer c.mu.Unlock()

	if elem, ok := c.cache[hash]; ok {
		c.lru.MoveToFront(elem)
		elem.Value.(*cacheEntry).vector = vector
		return
	}

	entry := &cacheEntry{hash: hash, vector: vector}
	elem := c.lru.PushFront(entry)
	c.cache[hash] = elem

	if c.lru.Len() > c.capacity {
		c.evict()
	}
}

func (c *LRUCache) evict() {
	elem := c.lru.Back()
	if elem != nil {
		c.lru.Remove(elem)
		entry := elem.Value.(*cacheEntry)
		delete(c.cache, entry.hash)
	}
}

func (c *LRUCache) Len() int {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.lru.Len()
}

func (c *LRUCache) Clear() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.cache = make(map[uint64]*list.Element)
	c.lru = list.New()
}
