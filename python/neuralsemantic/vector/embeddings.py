"""Embedding management for vector operations."""

import logging
import numpy as np
from typing import Dict, Any, Optional
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    SentenceTransformer = None

logger = logging.getLogger(__name__)


class EmbeddingManager:
    """Manages text embeddings with caching and optimization."""

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        self._embedding_cache: Dict[str, np.ndarray] = {}
        self._cache_hits = 0
        self._cache_misses = 0
        
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            logger.warning("sentence-transformers not available, using fallback embeddings")
            self.model = None
            self.enabled = False
        else:
            self._load_model()

    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)
            self.enabled = True
            logger.info("Embedding model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            self.model = None
            self.enabled = False

    def get_embedding(self, text: str) -> np.ndarray:
        """Get embedding for text with caching."""
        if not self.enabled:
            return self._fallback_embedding(text)
            
        # Check cache
        if text in self._embedding_cache:
            self._cache_hits += 1
            return self._embedding_cache[text]

        # Generate embedding
        try:
            embedding = self.model.encode(text, convert_to_tensor=False)
            self._embedding_cache[text] = embedding
            self._cache_misses += 1

            # Limit cache size to prevent memory issues
            if len(self._embedding_cache) > 10000:
                self._cleanup_cache()

            return embedding

        except Exception as e:
            logger.error(f"Failed to generate embedding: {e}")
            return self._fallback_embedding(text)

    def _fallback_embedding(self, text: str) -> np.ndarray:
        """Generate simple fallback embedding when model is not available."""
        # Simple character-based embedding as fallback
        # This is not ideal but allows the system to function
        chars = [ord(c) for c in text.lower()[:384]]  # Limit length
        
        # Pad or truncate to fixed size
        if len(chars) < 384:
            chars.extend([0] * (384 - len(chars)))
        else:
            chars = chars[:384]
        
        # Normalize
        embedding = np.array(chars, dtype=np.float32)
        norm = np.linalg.norm(embedding)
        if norm > 0:
            embedding = embedding / norm
            
        return embedding

    def _cleanup_cache(self) -> None:
        """Clean up embedding cache to manage memory."""
        # Remove half of the cache entries (oldest first)
        cache_items = list(self._embedding_cache.items())
        keep_count = len(cache_items) // 2
        
        self._embedding_cache = dict(cache_items[-keep_count:])
        logger.info(f"Cleaned embedding cache, kept {keep_count} entries")

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get embedding cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0
        
        return {
            "enabled": self.enabled,
            "model_name": self.model_name,
            "cache_size": len(self._embedding_cache),
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": hit_rate
        }

    def clear_cache(self) -> None:
        """Clear embedding cache."""
        self._embedding_cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
        logger.info("Embedding cache cleared")

    def batch_encode(self, texts: list[str]) -> np.ndarray:
        """Encode multiple texts efficiently."""
        if not self.enabled:
            return np.array([self._fallback_embedding(text) for text in texts])
            
        try:
            # Check cache for existing embeddings
            cached_embeddings = {}
            uncached_texts = []
            uncached_indices = []
            
            for i, text in enumerate(texts):
                if text in self._embedding_cache:
                    cached_embeddings[i] = self._embedding_cache[text]
                    self._cache_hits += 1
                else:
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # Generate embeddings for uncached texts
            if uncached_texts:
                new_embeddings = self.model.encode(uncached_texts, convert_to_tensor=False)
                self._cache_misses += len(uncached_texts)
                
                # Update cache
                for text, embedding in zip(uncached_texts, new_embeddings):
                    self._embedding_cache[text] = embedding
            
            # Combine cached and new embeddings
            result_embeddings = [None] * len(texts)
            
            # Add cached embeddings
            for i, embedding in cached_embeddings.items():
                result_embeddings[i] = embedding
            
            # Add new embeddings
            if uncached_texts:
                for i, embedding in zip(uncached_indices, new_embeddings):
                    result_embeddings[i] = embedding
            
            return np.array(result_embeddings)
            
        except Exception as e:
            logger.error(f"Batch encoding failed: {e}")
            return np.array([self._fallback_embedding(text) for text in texts])

    def similarity(self, text1: str, text2: str) -> float:
        """Calculate cosine similarity between two texts."""
        embedding1 = self.get_embedding(text1)
        embedding2 = self.get_embedding(text2)
        
        # Cosine similarity
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return dot_product / (norm1 * norm2)

    def find_most_similar(self, query_text: str, candidate_texts: list[str], 
                         top_k: int = 5) -> list[tuple[str, float]]:
        """Find most similar texts from candidates."""
        query_embedding = self.get_embedding(query_text)
        candidate_embeddings = self.batch_encode(candidate_texts)
        
        similarities = []
        for i, candidate_embedding in enumerate(candidate_embeddings):
            # Cosine similarity
            dot_product = np.dot(query_embedding, candidate_embedding)
            norm1 = np.linalg.norm(query_embedding)
            norm2 = np.linalg.norm(candidate_embedding)
            
            if norm1 > 0 and norm2 > 0:
                similarity = dot_product / (norm1 * norm2)
            else:
                similarity = 0.0
                
            similarities.append((candidate_texts[i], similarity))
        
        # Sort by similarity and return top k
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]