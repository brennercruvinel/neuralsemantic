"""ChromaDB-based vector storage for semantic similarity."""

import logging
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional

try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    chromadb = None

from ..core.types import Pattern, SimilarPattern
from ..core.exceptions import VectorStoreError
from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class VectorStore:
    """ChromaDB-based vector storage for semantic similarity."""

    def __init__(self, config):
        if not CHROMADB_AVAILABLE:
            logger.warning("ChromaDB not available, vector features disabled")
            self.enabled = False
            return
            
        self.config = config
        self.enabled = True
        self._initialize_client()
        self.embedding_manager = EmbeddingManager(config.model_name)

    def _initialize_client(self) -> None:
        """Initialize ChromaDB client."""
        try:
            # Ensure directory exists
            persist_dir = Path(self.config.persist_directory)
            persist_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize client
            self.client = chromadb.PersistentClient(
                path=str(persist_dir),
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )
            
            # Get or create collection
            try:
                # Try to get existing collection first
                self.collection = self.client.get_collection(name="compression_patterns")
            except:
                # Create new collection with proper metadata format
                self.collection = self.client.create_collection(
                    name="compression_patterns",
                    metadata={"hnsw_space": "cosine"}  # Simplified metadata
                )
            
            logger.info(f"Vector store initialized at {self.config.persist_directory}")
            
        except Exception as e:
            logger.error(f"Failed to initialize vector store: {e}")
            self.enabled = False
            raise VectorStoreError(f"Vector store initialization failed: {e}")

    def add_pattern(self, pattern: Pattern) -> None:
        """Add pattern to vector store with metadata."""
        if not self.enabled:
            return
            
        try:
            # Generate embedding
            embedding = self.embedding_manager.get_embedding(pattern.original)
            
            # Prepare metadata
            metadata = {
                "pattern_id": pattern.id or 0,
                "compressed": pattern.compressed,
                "pattern_type": pattern.pattern_type.value,
                "domain": pattern.domain,
                "priority": pattern.priority,
                "compression_ratio": len(pattern.compressed) / len(pattern.original),
                "language": pattern.language
            }

            # Add to collection
            self.collection.add(
                embeddings=[embedding.tolist()],
                documents=[pattern.original],
                metadatas=[metadata],
                ids=[f"pattern_{pattern.id or hash(pattern.original)}"]
            )

        except Exception as e:
            logger.error(f"Failed to add pattern to vector store: {e}")

    def find_similar_patterns(self, text: str, n_results: int = 5,
                            threshold: float = 0.8, domain: Optional[str] = None) -> List[SimilarPattern]:
        """Find semantically similar patterns."""
        if not self.enabled:
            return []
            
        try:
            # Generate query embedding
            query_embedding = self.embedding_manager.get_embedding(text)

            # Prepare where clause for filtering
            where_clause = {}
            if domain:
                where_clause["domain"] = domain

            # Query vector store
            results = self.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n_results * 2,  # Get more to filter
                where=where_clause if where_clause else None,
                include=["metadatas", "documents", "distances"]
            )

            # Process results
            similar_patterns = []
            if results['distances'] and results['distances'][0]:
                for i, distance in enumerate(results['distances'][0]):
                    similarity = 1 - distance

                    if similarity >= threshold:
                        metadata = results['metadatas'][0][i]
                        similar_patterns.append(SimilarPattern(
                            original=results['documents'][0][i],
                            compressed=metadata['compressed'],
                            similarity=similarity,
                            pattern_type=metadata['pattern_type'],
                            domain=metadata['domain'],
                            priority=metadata['priority'],
                            confidence=similarity * metadata['priority'] / 1000
                        ))

            # Sort by similarity and return top results
            similar_patterns.sort(key=lambda x: x.similarity, reverse=True)
            return similar_patterns[:n_results]

        except Exception as e:
            logger.error(f"Vector similarity search failed: {e}")
            return []

    def bulk_add_patterns(self, patterns: List[Pattern]) -> None:
        """Efficiently add multiple patterns."""
        if not self.enabled or not patterns:
            return

        try:
            embeddings = []
            documents = []
            metadatas = []
            ids = []

            for pattern in patterns:
                embedding = self.embedding_manager.get_embedding(pattern.original)
                embeddings.append(embedding.tolist())
                documents.append(pattern.original)
                metadatas.append({
                    "pattern_id": pattern.id or 0,
                    "compressed": pattern.compressed,
                    "pattern_type": pattern.pattern_type.value,
                    "domain": pattern.domain,
                    "priority": pattern.priority,
                    "compression_ratio": len(pattern.compressed) / len(pattern.original),
                    "language": pattern.language
                })
                ids.append(f"pattern_{pattern.id or hash(pattern.original)}")

            # Batch add to collection
            self.collection.add(
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )

            logger.info(f"Added {len(patterns)} patterns to vector store")

        except Exception as e:
            logger.error(f"Bulk add to vector store failed: {e}")

    def get_collection_stats(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        if not self.enabled:
            return {"enabled": False}
            
        try:
            count = self.collection.count()
            return {
                "enabled": True,
                "total_patterns": count,
                "collection_name": self.collection.name,
                "model_name": self.embedding_manager.model_name
            }
        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {"enabled": True, "error": str(e)}

    def clear_collection(self) -> None:
        """Clear all patterns from vector store."""
        if not self.enabled:
            return
            
        try:
            # Delete and recreate collection
            self.client.delete_collection("compression_patterns")
            self.collection = self.client.get_or_create_collection(
                name="compression_patterns",
                metadata={
                    "hnsw:space": "cosine",
                    "hnsw:M": 16,
                    "hnsw:ef_construction": 200
                }
            )
            logger.info("Vector store collection cleared")
        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")

    def search_by_metadata(self, metadata_filter: Dict[str, Any], 
                          n_results: int = 20) -> List[Dict[str, Any]]:
        """Search patterns by metadata."""
        if not self.enabled:
            return []
            
        try:
            results = self.collection.get(
                where=metadata_filter,
                limit=n_results,
                include=["metadatas", "documents"]
            )
            
            patterns = []
            if results['documents']:
                for i, doc in enumerate(results['documents']):
                    patterns.append({
                        "original": doc,
                        "metadata": results['metadatas'][i]
                    })
            
            return patterns
            
        except Exception as e:
            logger.error(f"Metadata search failed: {e}")
            return []

    def update_pattern_metadata(self, pattern_id: str, metadata: Dict[str, Any]) -> bool:
        """Update pattern metadata."""
        if not self.enabled:
            return False
            
        try:
            self.collection.update(
                ids=[pattern_id],
                metadatas=[metadata]
            )
            return True
        except Exception as e:
            logger.error(f"Failed to update pattern metadata: {e}")
            return False

    def delete_pattern(self, pattern_id: str) -> bool:
        """Delete pattern from vector store."""
        if not self.enabled:
            return True  # No-op if disabled
            
        try:
            self.collection.delete(ids=[pattern_id])
            return True
        except Exception as e:
            logger.error(f"Failed to delete pattern from vector store: {e}")
            return False