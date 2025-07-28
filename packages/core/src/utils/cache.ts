/**
 * Cache Manager for Neural Semantic Compiler
 */

export class CacheManager {
  private cache: Map<string, CacheEntry>;
  private maxSize: number;
  private ttlMs: number;

  constructor(maxSize: number = 1000, ttlSeconds: number = 3600) {
    this.cache = new Map();
    this.maxSize = maxSize;
    this.ttlMs = ttlSeconds * 1000;
  }

  /**
   * Get value from cache
   */
  get<T>(key: string): T | null {
    const entry = this.cache.get(key);
    
    if (!entry) {
      return null;
    }

    // Check if expired
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return null;
    }

    // Update access time
    entry.lastAccessed = Date.now();
    
    return entry.value as T;
  }

  /**
   * Set value in cache
   */
  set<T>(key: string, value: T, customTtlSeconds?: number): void {
    const ttl = customTtlSeconds ? customTtlSeconds * 1000 : this.ttlMs;
    const now = Date.now();

    // Remove old entry if exists
    if (this.cache.has(key)) {
      this.cache.delete(key);
    }

    // Check size limit
    if (this.cache.size >= this.maxSize) {
      this.evictLeastRecentlyUsed();
    }

    const entry: CacheEntry = {
      value,
      createdAt: now,
      lastAccessed: now,
      expiresAt: now + ttl
    };

    this.cache.set(key, entry);
  }

  /**
   * Check if key exists in cache
   */
  has(key: string): boolean {
    const entry = this.cache.get(key);
    
    if (!entry) {
      return false;
    }

    // Check if expired
    if (Date.now() > entry.expiresAt) {
      this.cache.delete(key);
      return false;
    }

    return true;
  }

  /**
   * Delete entry from cache
   */
  delete(key: string): boolean {
    return this.cache.delete(key);
  }

  /**
   * Clear all cache entries
   */
  clear(): void {
    this.cache.clear();
  }

  /**
   * Get cache statistics
   */
  getStats(): {
    size: number;
    maxSize: number;
    hitRate: number;
    memoryUsage: number;
    oldestEntry: number;
    newestEntry: number;
  } {
    const now = Date.now();
    let oldestTime = now;
    let newestTime = 0;
    let totalHits = 0;
    let totalRequests = 0;

    for (const entry of this.cache.values()) {
      if (entry.createdAt < oldestTime) {
        oldestTime = entry.createdAt;
      }
      if (entry.createdAt > newestTime) {
        newestTime = entry.createdAt;
      }
    }

    // Estimate memory usage (rough calculation)
    const memoryUsage = this.cache.size * 100; // Approximate bytes per entry

    return {
      size: this.cache.size,
      maxSize: this.maxSize,
      hitRate: totalRequests > 0 ? totalHits / totalRequests : 0,
      memoryUsage,
      oldestEntry: now - oldestTime,
      newestEntry: now - newestTime
    };
  }

  /**
   * Remove expired entries
   */
  cleanupExpired(): number {
    const now = Date.now();
    let removed = 0;

    for (const [key, entry] of this.cache) {
      if (now > entry.expiresAt) {
        this.cache.delete(key);
        removed++;
      }
    }

    return removed;
  }

  /**
   * Get all keys in cache
   */
  keys(): string[] {
    this.cleanupExpired();
    return Array.from(this.cache.keys());
  }

  /**
   * Get cache size
   */
  size(): number {
    this.cleanupExpired();
    return this.cache.size;
  }

  private evictLeastRecentlyUsed(): void {
    let lruKey: string | null = null;
    let lruTime = Date.now();

    for (const [key, entry] of this.cache) {
      if (entry.lastAccessed < lruTime) {
        lruTime = entry.lastAccessed;
        lruKey = key;
      }
    }

    if (lruKey) {
      this.cache.delete(lruKey);
    }
  }
}

interface CacheEntry {
  value: any;
  createdAt: number;
  lastAccessed: number;
  expiresAt: number;
}