"""
LRU Image Cache for storing fetched images with size limit.
Extracted from duplicated code in server.py, server_v2.py, server_medical.py
"""

from collections import OrderedDict
import hashlib


class LRUImageCache:
    """
    LRU cache for storing fetched images with size limit.
    Automatically evicts oldest items when size limit is exceeded.
    """
    
    def __init__(self, max_size_bytes):
        """
        Initialize LRU cache.
        
        Args:
            max_size_bytes: Maximum cache size in bytes
        """
        self.max_size_bytes = max_size_bytes
        self.current_size = 0
        self.cache = OrderedDict()

    def get(self, key):
        """
        Get item from cache, move to end (most recently used).
        
        Args:
            key: Cache key
            
        Returns:
            Cached value or None if not found
        """
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key, value):
        """
        Add item to cache, evict oldest if over size limit.
        
        Args:
            key: Cache key
            value: Value to cache (bytes)
        """
        value_size = len(value)

        # If single item is larger than cache, don't cache it
        if value_size > self.max_size_bytes:
            return

        # Evict oldest items until we have space
        while self.current_size + value_size > self.max_size_bytes and self.cache:
            oldest_key, oldest_value = self.cache.popitem(last=False)
            self.current_size -= len(oldest_value)

        # Add new item
        if key in self.cache:
            self.current_size -= len(self.cache[key])
        self.cache[key] = value
        self.cache.move_to_end(key)
        self.current_size += value_size

    def clear(self):
        """Clear all cached items."""
        self.cache.clear()
        self.current_size = 0

    def stats(self):
        """
        Return cache statistics.
        
        Returns:
            dict: Cache statistics including entries, size, and max size
        """
        return {
            'entries': len(self.cache),
            'size_mb': round(self.current_size / (1024 * 1024), 2),
            'max_size_mb': round(self.max_size_bytes / (1024 * 1024), 2),
            'utilization_pct': round((self.current_size / self.max_size_bytes) * 100, 1)
        }

    @staticmethod
    def make_cache_key(url):
        """
        Create a cache key from a URL.
        
        Args:
            url: URL string
            
        Returns:
            str: MD5 hash of URL
        """
        return hashlib.md5(url.encode()).hexdigest()
