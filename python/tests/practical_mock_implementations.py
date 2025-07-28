"""
Practical Mock Implementations for Neural Semantic Compiler

This file provides ready-to-use mock implementations for all major components
of the Neural Semantic Compiler. These can be used immediately for development
and gradually replaced with real implementations.
"""

import pytest
from unittest.mock import Mock, MagicMock, patch, AsyncMock
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import json
import time
import random
from datetime import datetime


# ============================================================================
# DATA STRUCTURES AND TYPES
# ============================================================================

@dataclass
class MockCompressionResult:
    """Mock compression result with realistic data structure."""
    original_text: str
    compressed_text: str
    compression_ratio: float
    quality_score: float
    original_tokens: int
    compressed_tokens: int
    engine_used: str
    processing_time_ms: float
    patterns_applied: List[str]
    context: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MockPattern:
    """Mock pattern structure."""
    id: int
    original: str
    compressed: str
    type: str  # 'word', 'phrase', 'compound'
    domain: str
    priority: int
    quality: float
    usage_count: int = 0
    created_at: Optional[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now().isoformat()


@dataclass
class MockHealthStatus:
    """Mock system health status."""
    overall_status: str  # 'healthy', 'warning', 'error'
    components: Dict[str, str]
    last_check: str
    uptime_seconds: float


# ============================================================================
# CORE COMPONENT MOCKS
# ============================================================================

class MockNeuralSemanticCompiler:
    """
    Complete mock implementation of the Neural Semantic Compiler.
    Provides realistic behavior for development and testing.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.patterns = self._initialize_mock_patterns()
        self.compression_history = []
        self.start_time = time.time()
        
        # Mock component availability
        self.vector_store_available = config.get('vector_store_available', True)
        self.database_available = config.get('database_available', True)
        
    def _initialize_mock_patterns(self) -> List[MockPattern]:
        """Initialize with realistic mock patterns."""
        return [
            MockPattern(1, "machine learning", "ML", "compound", "ai", 900, 0.95),
            MockPattern(2, "artificial intelligence", "AI", "compound", "ai", 950, 0.98),
            MockPattern(3, "user interface", "UI", "compound", "web", 800, 0.90),
            MockPattern(4, "application programming interface", "API", "compound", "web", 850, 0.92),
            MockPattern(5, "database management system", "DBMS", "compound", "data", 750, 0.88),
            MockPattern(6, "continuous integration", "CI", "compound", "devops", 700, 0.85),
            MockPattern(7, "production ready", "prod-ready", "phrase", "general", 600, 0.80),
            MockPattern(8, "implementation", "impl", "word", "general", 500, 0.75),
            MockPattern(9, "configuration", "config", "word", "general", 550, 0.78),
            MockPattern(10, "authentication", "auth", "word", "security", 650, 0.82),
        ]
    
    def compress(self, text: str, level: str = "balanced", domain: Optional[str] = None, 
                preserve_code: bool = False) -> MockCompressionResult:
        """
        Mock compression with realistic behavior based on patterns and settings.
        """
        start_time = time.time()
        
        # Simulate different behavior based on parameters
        if preserve_code and ('def ' in text or 'function ' in text or '{' in text):
            # Minimal compression for code
            compressed_text = text.replace('  ', ' ').strip()
            compression_ratio = len(compressed_text) / len(text)
            quality_score = 0.95  # High quality for preserved code
            patterns_applied = ["code_preservation"]
        else:
            # Apply mock pattern matching
            compressed_text, patterns_applied = self._apply_mock_patterns(text, domain)
            
            # Adjust compression based on level
            compression_factor = {
                "light": 0.9,
                "balanced": 0.75,
                "aggressive": 0.6
            }.get(level, 0.75)
            
            if len(compressed_text) / len(text) > compression_factor:
                # Apply additional compression to meet target
                additional_compression = self._apply_additional_compression(compressed_text, compression_factor)
                compressed_text = additional_compression
            
            compression_ratio = len(compressed_text) / len(text)
            quality_score = self._calculate_mock_quality_score(text, compressed_text, patterns_applied)
        
        # Choose engine based on availability and text characteristics
        engine_used = self._select_mock_engine(text, level)
        
        processing_time = (time.time() - start_time) * 1000  # Convert to ms
        
        result = MockCompressionResult(
            original_text=text,
            compressed_text=compressed_text,
            compression_ratio=compression_ratio,
            quality_score=quality_score,
            original_tokens=len(text.split()),
            compressed_tokens=len(compressed_text.split()),
            engine_used=engine_used,
            processing_time_ms=processing_time,
            patterns_applied=patterns_applied,
            context={"level": level, "domain": domain, "preserve_code": preserve_code}
        )
        
        self.compression_history.append(result)
        return result
    
    def _apply_mock_patterns(self, text: str, domain: Optional[str] = None) -> tuple[str, List[str]]:
        """Apply mock pattern matching to text."""
        compressed_text = text
        patterns_applied = []
        
        # Filter patterns by domain if specified
        relevant_patterns = [p for p in self.patterns if not domain or p.domain == domain or p.domain == "general"]
        
        # Sort by priority (higher first)
        relevant_patterns.sort(key=lambda x: x.priority, reverse=True)
        
        for pattern in relevant_patterns:
            if pattern.original.lower() in text.lower():
                # Apply pattern with some probability based on quality
                if random.random() < pattern.quality:
                    compressed_text = compressed_text.replace(pattern.original, pattern.compressed)
                    patterns_applied.append(f"{pattern.original} -> {pattern.compressed}")
                    pattern.usage_count += 1
        
        return compressed_text, patterns_applied
    
    def _apply_additional_compression(self, text: str, target_ratio: float) -> str:
        """Apply additional mock compression techniques."""
        # Remove extra spaces
        text = ' '.join(text.split())
        
        # Remove common articles if still too long
        if len(text) / len(self.compression_history[-1].original_text if self.compression_history else text) > target_ratio:
            text = text.replace(' the ', ' ').replace(' a ', ' ').replace(' an ', ' ')
        
        # Remove common prepositions if still too long  
        if len(text) / len(self.compression_history[-1].original_text if self.compression_history else text) > target_ratio:
            text = text.replace(' of ', ' ').replace(' in ', ' ').replace(' on ', ' ')
        
        return text.strip()
    
    def _calculate_mock_quality_score(self, original: str, compressed: str, patterns: List[str]) -> float:
        """Calculate realistic mock quality score."""
        base_score = 0.7
        
        # Bonus for pattern usage
        pattern_bonus = min(0.2, len(patterns) * 0.05)
        
        # Penalty for excessive compression
        compression_ratio = len(compressed) / len(original)
        if compression_ratio < 0.3:
            compression_penalty = (0.3 - compression_ratio) * 0.5
        else:
            compression_penalty = 0
        
        # Bonus for preserving important words
        important_words = ['class', 'function', 'method', 'variable', 'parameter']
        preservation_bonus = sum(0.02 for word in important_words if word in compressed) 
        
        final_score = base_score + pattern_bonus - compression_penalty + preservation_bonus
        return min(1.0, max(0.1, final_score))
    
    def _select_mock_engine(self, text: str, level: str) -> str:
        """Select appropriate mock engine based on conditions."""
        if not self.vector_store_available:
            return "FallbackEngine"
        
        if len(text.split()) < 5:
            return "SimpleEngine"
        
        if level == "aggressive":
            return "ExtremeEngine"
        elif any(domain_word in text.lower() for domain_word in ['ai', 'ml', 'machine', 'learning']):
            return "SemanticEngine"
        else:
            return "HybridEngine"
    
    def add_pattern(self, original: str, compressed: str, pattern_type: str = "word", 
                   domain: str = "general", priority: int = 500) -> bool:
        """Add a new pattern to the mock pattern store."""
        if not original or not compressed:
            return False
        
        if len(compressed) > len(original):
            return False  # Compressed should be shorter
        
        new_id = max(p.id for p in self.patterns) + 1 if self.patterns else 1
        new_pattern = MockPattern(
            id=new_id,
            original=original,
            compressed=compressed,
            type=pattern_type,
            domain=domain,
            priority=priority,
            quality=0.8  # Default quality
        )
        
        self.patterns.append(new_pattern)
        return True
    
    def get_patterns(self, domain: Optional[str] = None, pattern_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get patterns with optional filtering."""
        filtered_patterns = self.patterns
        
        if domain:
            filtered_patterns = [p for p in filtered_patterns if p.domain == domain]
        
        if pattern_type:
            filtered_patterns = [p for p in filtered_patterns if p.type == pattern_type]
        
        return [asdict(p) for p in filtered_patterns]
    
    def search_patterns(self, query: str) -> List[Dict[str, Any]]:
        """Search patterns by query."""
        query_lower = query.lower()
        matching_patterns = [
            p for p in self.patterns 
            if query_lower in p.original.lower() or query_lower in p.compressed.lower()
        ]
        return [asdict(p) for p in matching_patterns]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive mock statistics."""
        total_compressions = len(self.compression_history)
        
        if total_compressions == 0:
            return {
                "session": {"compressions": 0, "total_input_tokens": 0, "total_output_tokens": 0},
                "patterns": {"total": len(self.patterns), "most_used": None},
                "engines": {"usage": {}},
                "quality": {"average": 0.0, "distribution": {}}
            }
        
        total_input_tokens = sum(r.original_tokens for r in self.compression_history)
        total_output_tokens = sum(r.compressed_tokens for r in self.compression_history)
        
        # Engine usage statistics
        engine_usage = {}
        for result in self.compression_history:
            engine_usage[result.engine_used] = engine_usage.get(result.engine_used, 0) + 1
        
        # Quality distribution
        quality_scores = [r.quality_score for r in self.compression_history]
        avg_quality = sum(quality_scores) / len(quality_scores)
        
        # Most used pattern
        pattern_usage = {p.original: p.usage_count for p in self.patterns}
        most_used_pattern = max(pattern_usage.items(), key=lambda x: x[1]) if pattern_usage else None
        
        return {
            "session": {
                "compressions": total_compressions,
                "total_input_tokens": total_input_tokens,
                "total_output_tokens": total_output_tokens,
                "average_compression_ratio": sum(r.compression_ratio for r in self.compression_history) / total_compressions,
                "total_processing_time_ms": sum(r.processing_time_ms for r in self.compression_history)
            },
            "patterns": {
                "total": len(self.patterns),
                "most_used": most_used_pattern[0] if most_used_pattern else None,
                "usage_distribution": pattern_usage
            },
            "engines": {
                "usage": engine_usage,
                "most_used": max(engine_usage.items(), key=lambda x: x[1])[0] if engine_usage else None
            },
            "quality": {
                "average": avg_quality,
                "min": min(quality_scores),
                "max": max(quality_scores),
                "distribution": {
                    "excellent": len([q for q in quality_scores if q >= 0.9]),
                    "good": len([q for q in quality_scores if 0.7 <= q < 0.9]),
                    "fair": len([q for q in quality_scores if 0.5 <= q < 0.7]),
                    "poor": len([q for q in quality_scores if q < 0.5])
                }
            }
        }
    
    def validate_system_health(self) -> MockHealthStatus:
        """Perform mock system health check."""
        components = {
            "pattern_manager": "healthy" if self.database_available else "error",
            "vector_store": "healthy" if self.vector_store_available else "warning",
            "engines": "healthy",
            "database": "healthy" if self.database_available else "error"
        }
        
        # Determine overall status
        if any(status == "error" for status in components.values()):
            overall_status = "error"
        elif any(status == "warning" for status in components.values()):
            overall_status = "warning"
        else:
            overall_status = "healthy"
        
        return MockHealthStatus(
            overall_status=overall_status,
            components=components,
            last_check=datetime.now().isoformat(),
            uptime_seconds=time.time() - self.start_time
        )
    
    def explain_compression(self, text: str) -> Dict[str, Any]:
        """Provide detailed explanation of compression process."""
        result = self.compress(text)
        
        return {
            "input": {
                "text": text,
                "tokens": result.original_tokens,
                "characters": len(text)
            },
            "output": {
                "text": result.compressed_text,
                "tokens": result.compressed_tokens,
                "characters": len(result.compressed_text)
            },
            "compression": {
                "ratio": result.compression_ratio,
                "token_reduction": result.original_tokens - result.compressed_tokens,
                "character_reduction": len(text) - len(result.compressed_text)
            },
            "engine": {
                "used": result.engine_used,
                "processing_time_ms": result.processing_time_ms
            },
            "patterns_applied": [
                {
                    "transformation": pattern,
                    "type": "pattern_match"
                } for pattern in result.patterns_applied
            ],
            "quality": {
                "score": result.quality_score,
                "assessment": (
                    "excellent" if result.quality_score >= 0.9 else
                    "good" if result.quality_score >= 0.7 else
                    "fair" if result.quality_score >= 0.5 else
                    "poor"
                )
            }
        }
    
    def compare_engines(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Compare different engine results for the same text."""
        engines = ["SemanticEngine", "HybridEngine", "ExtremeEngine", "FallbackEngine"]
        results = {}
        
        for engine in engines:
            try:
                # Simulate different engine behaviors
                if engine == "SemanticEngine":
                    result = self._simulate_semantic_engine(text)
                elif engine == "HybridEngine": 
                    result = self._simulate_hybrid_engine(text)
                elif engine == "ExtremeEngine":
                    result = self._simulate_extreme_engine(text)
                else:  # FallbackEngine
                    result = self._simulate_fallback_engine(text)
                
                results[engine] = {
                    "compression_ratio": result["compression_ratio"],
                    "quality_score": result["quality_score"],
                    "processing_time_ms": result["processing_time_ms"],
                    "patterns_applied": result["patterns_applied"]
                }
            except Exception as e:
                results[engine] = {"error": str(e)}
        
        return results
    
    def _simulate_semantic_engine(self, text: str) -> Dict[str, Any]:
        """Simulate semantic engine with high quality, moderate compression."""
        compressed_text, patterns = self._apply_mock_patterns(text, None)
        return {
            "compressed_text": compressed_text,
            "compression_ratio": len(compressed_text) / len(text),
            "quality_score": 0.9,
            "processing_time_ms": 25.0,
            "patterns_applied": patterns
        }
    
    def _simulate_hybrid_engine(self, text: str) -> Dict[str, Any]:
        """Simulate hybrid engine with balanced compression and quality."""
        compressed_text, patterns = self._apply_mock_patterns(text, None)
        compressed_text = self._apply_additional_compression(compressed_text, 0.75)
        return {
            "compressed_text": compressed_text,
            "compression_ratio": len(compressed_text) / len(text),
            "quality_score": 0.8,
            "processing_time_ms": 15.0,
            "patterns_applied": patterns
        }
    
    def _simulate_extreme_engine(self, text: str) -> Dict[str, Any]:
        """Simulate extreme engine with high compression, lower quality."""
        compressed_text, patterns = self._apply_mock_patterns(text, None)
        compressed_text = self._apply_additional_compression(compressed_text, 0.5)
        return {
            "compressed_text": compressed_text,
            "compression_ratio": len(compressed_text) / len(text),
            "quality_score": 0.6,
            "processing_time_ms": 30.0,
            "patterns_applied": patterns + ["extreme_compression"]
        }
    
    def _simulate_fallback_engine(self, text: str) -> Dict[str, Any]:
        """Simulate fallback engine with minimal compression."""
        compressed_text = ' '.join(text.split())  # Just normalize whitespace
        return {
            "compressed_text": compressed_text,
            "compression_ratio": len(compressed_text) / len(text),
            "quality_score": 0.95,
            "processing_time_ms": 2.0,
            "patterns_applied": ["whitespace_normalization"]
        }
    
    def get_session_report(self) -> str:
        """Generate a comprehensive session report."""
        stats = self.get_statistics()
        health = self.validate_system_health()
        
        report = f"""
Neural Semantic Compiler - Session Report
========================================

Session Statistics:
- Total compressions: {stats['session']['compressions']}
- Total input tokens: {stats['session']['total_input_tokens']}
- Total output tokens: {stats['session']['total_output_tokens']}
- Average compression ratio: {stats['session'].get('average_compression_ratio', 0):.2f}
- Total processing time: {stats['session']['total_processing_time_ms']:.1f}ms

Pattern Usage:
- Total patterns available: {stats['patterns']['total']}
- Most used pattern: {stats['patterns']['most_used'] or 'None'}

Engine Performance:
- Most used engine: {stats['engines'].get('most_used', 'None')}
- Engine usage: {stats['engines']['usage']}

Quality Metrics:
- Average quality score: {stats['quality']['average']:.2f}
- Quality distribution:
  * Excellent (≥0.9): {stats['quality']['distribution']['excellent']}
  * Good (0.7-0.9): {stats['quality']['distribution']['good']}
  * Fair (0.5-0.7): {stats['quality']['distribution']['fair']}
  * Poor (<0.5): {stats['quality']['distribution']['poor']}

System Health:
- Overall status: {health.overall_status}
- Uptime: {health.uptime_seconds:.1f} seconds
- Component status: {health.components}
        """.strip()
        
        return report
    
    def close(self):
        """Clean up resources (mock implementation)."""
        # In real implementation, this would close database connections, etc.
        pass
    
    @classmethod
    def create_default(cls) -> 'MockNeuralSemanticCompiler':
        """Create compiler with default configuration."""
        return cls({
            'vector_store_available': True,
            'database_available': True
        })


# ============================================================================
# SPECIALIZED MOCK COMPONENTS
# ============================================================================

class MockVectorStore:
    """Mock vector store for semantic similarity operations."""
    
    def __init__(self, available: bool = True):
        self.available = available
        self.patterns = {}
        self.embeddings_cache = {}
    
    def is_available(self) -> bool:
        return self.available
    
    def add_pattern(self, original: str, compressed: str, metadata: Optional[Dict] = None) -> bool:
        if not self.available:
            raise RuntimeError("Vector store unavailable")
        
        self.patterns[original] = {
            'compressed': compressed,
            'metadata': metadata or {},
            'embedding': self._mock_embedding(original)
        }
        return True
    
    def search_similar(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        if not self.available:
            return []
        
        query_embedding = self._mock_embedding(query)
        similarities = []
        
        for original, data in self.patterns.items():
            similarity = self._calculate_mock_similarity(query_embedding, data['embedding'])
            similarities.append({
                'pattern': original,
                'compressed': data['compressed'],
                'similarity': similarity,
                'metadata': data['metadata']
            })
        
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return similarities[:limit]
    
    def _mock_embedding(self, text: str) -> List[float]:
        """Generate mock embedding vector."""
        # Use hash for deterministic but varied embeddings
        hash_val = hash(text)
        return [(hash_val >> i) % 100 / 100.0 for i in range(10)]
    
    def _calculate_mock_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Calculate mock cosine similarity."""
        dot_product = sum(a * b for a, b in zip(embedding1, embedding2))
        magnitude1 = sum(a * a for a in embedding1) ** 0.5
        magnitude2 = sum(b * b for b in embedding2) ** 0.5
        
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        return dot_product / (magnitude1 * magnitude2)


class MockPatternManager:
    """Mock pattern manager for database operations."""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or ":memory:"
        self.patterns = {}
        self.next_id = 1
    
    def get_patterns(self, domain: Optional[str] = None, pattern_type: Optional[str] = None) -> List[Dict[str, Any]]:
        filtered = list(self.patterns.values())
        
        if domain:
            filtered = [p for p in filtered if p.get('domain') == domain]
        
        if pattern_type:
            filtered = [p for p in filtered if p.get('type') == pattern_type]
        
        return filtered
    
    def add_pattern(self, original: str, compressed: str, pattern_type: str, domain: str, priority: int = 500) -> bool:
        if not original or not compressed:
            return False
        
        pattern_id = self.next_id
        self.next_id += 1
        
        self.patterns[pattern_id] = {
            'id': pattern_id,
            'original': original,
            'compressed': compressed,
            'type': pattern_type,
            'domain': domain,
            'priority': priority,
            'quality': 0.8,
            'usage_count': 0,
            'created_at': datetime.now().isoformat()
        }
        
        return True
    
    def search_patterns(self, query: str) -> List[Dict[str, Any]]:
        query_lower = query.lower()
        return [
            pattern for pattern in self.patterns.values()
            if query_lower in pattern['original'].lower() or query_lower in pattern['compressed'].lower()
        ]
    
    def update_pattern_usage(self, pattern_id: int) -> bool:
        if pattern_id in self.patterns:
            self.patterns[pattern_id]['usage_count'] += 1
            return True
        return False


class MockQualityScorer:
    """Mock quality scorer for compression evaluation."""
    
    def calculate_score(self, original: str, compressed: str, patterns_applied: List[str]) -> float:
        # Base score
        base_score = 0.7
        
        # Length-based penalty/bonus
        compression_ratio = len(compressed) / len(original)
        if compression_ratio > 0.8:
            length_factor = -0.1  # Penalty for low compression
        elif compression_ratio < 0.3:
            length_factor = -0.2  # Penalty for excessive compression
        else:
            length_factor = 0.1   # Bonus for good compression
        
        # Pattern application bonus
        pattern_bonus = min(0.2, len(patterns_applied) * 0.05)
        
        # Semantic preservation (mock check)
        semantic_score = self._mock_semantic_preservation(original, compressed)
        
        final_score = base_score + length_factor + pattern_bonus + semantic_score
        return max(0.1, min(1.0, final_score))
    
    def _mock_semantic_preservation(self, original: str, compressed: str) -> float:
        """Mock semantic preservation scoring."""
        # Simple heuristic: check if important words are preserved
        original_words = set(original.lower().split())
        compressed_words = set(compressed.lower().split())
        
        important_words = original_words.intersection({
            'class', 'function', 'method', 'variable', 'parameter', 
            'database', 'api', 'user', 'system', 'application'
        })
        
        preserved_important = important_words.intersection(compressed_words)
        
        if not important_words:
            return 0.0
        
        preservation_ratio = len(preserved_important) / len(important_words)
        return preservation_ratio * 0.2  # Max 0.2 bonus


# ============================================================================
# ASYNC MOCK IMPLEMENTATIONS  
# ============================================================================

class MockAsyncCompiler:
    """Mock async version of the Neural Semantic Compiler."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.sync_compiler = MockNeuralSemanticCompiler(config)
    
    async def compress_async(self, text: str, level: str = "balanced") -> MockCompressionResult:
        """Async compression simulation."""
        # Simulate async delay
        import asyncio
        await asyncio.sleep(0.01)  # 10ms delay
        
        return self.sync_compiler.compress(text, level)
    
    async def compress_batch_async(self, texts: List[str], level: str = "balanced") -> List[MockCompressionResult]:
        """Async batch compression."""
        import asyncio
        
        # Simulate parallel processing
        tasks = [self.compress_async(text, level) for text in texts]
        return await asyncio.gather(*tasks)
    
    async def add_pattern_async(self, original: str, compressed: str, pattern_type: str = "word") -> bool:
        """Async pattern addition."""
        import asyncio
        await asyncio.sleep(0.005)  # 5ms delay
        
        return self.sync_compiler.add_pattern(original, compressed, pattern_type)


# ============================================================================
# INTEGRATION HELPERS
# ============================================================================

class MockTestEnvironment:
    """Helper class for setting up comprehensive mock test environments."""
    
    def __init__(self):
        self.compiler = None
        self.vector_store = None
        self.pattern_manager = None
        
    def setup_complete_mock_environment(self) -> MockNeuralSemanticCompiler:
        """Set up a complete mock environment for testing."""
        config = {
            'vector_store_available': True,
            'database_available': True,
            'enable_logging': False
        }
        
        self.compiler = MockNeuralSemanticCompiler(config)
        self.vector_store = MockVectorStore(available=True)
        self.pattern_manager = MockPatternManager()
        
        # Pre-populate with realistic test data
        self._populate_test_data()
        
        return self.compiler
    
    def setup_degraded_mock_environment(self) -> MockNeuralSemanticCompiler:
        """Set up mock environment with some services unavailable."""
        config = {
            'vector_store_available': False,  # Simulate vector store issues
            'database_available': True,
            'enable_logging': False
        }
        
        self.compiler = MockNeuralSemanticCompiler(config)
        self.vector_store = MockVectorStore(available=False)
        self.pattern_manager = MockPatternManager()
        
        return self.compiler
    
    def _populate_test_data(self):
        """Populate mock services with realistic test data."""
        if self.compiler:
            # Add some test patterns
            test_patterns = [
                ("React application", "React app", "compound", "web"),
                ("machine learning model", "ML model", "compound", "ai"),
                ("user authentication", "user auth", "compound", "security"),
                ("database connection", "DB conn", "compound", "data"),
                ("production environment", "prod env", "compound", "devops")
            ]
            
            for original, compressed, ptype, domain in test_patterns:
                self.compiler.add_pattern(original, compressed, ptype, domain)
        
        if self.vector_store and self.vector_store.is_available():
            # Add semantic patterns to vector store
            vector_patterns = [
                ("neural network", "NN", {"domain": "ai", "confidence": 0.95}),
                ("artificial intelligence", "AI", {"domain": "ai", "confidence": 0.98}),
                ("continuous integration", "CI", {"domain": "devops", "confidence": 0.90}),
                ("user interface", "UI", {"domain": "web", "confidence": 0.92})
            ]
            
            for original, compressed, metadata in vector_patterns:
                self.vector_store.add_pattern(original, compressed, metadata)


# ============================================================================
# USAGE EXAMPLES AND TESTS
# ============================================================================

def example_usage():
    """Example of how to use the mock implementations."""
    
    print("=== Neural Semantic Compiler Mock Implementation Examples ===\n")
    
    # 1. Basic mock compiler usage
    print("1. Basic Mock Compiler Usage:")
    compiler = MockNeuralSemanticCompiler.create_default()
    
    result = compiler.compress("Build a production ready React application with user authentication")
    print(f"Original: {result.original_text}")
    print(f"Compressed: {result.compressed_text}")
    print(f"Ratio: {result.compression_ratio:.2f}")
    print(f"Quality: {result.quality_score:.2f}")
    print(f"Engine: {result.engine_used}")
    print(f"Patterns: {result.patterns_applied}")
    
    # 2. Pattern management
    print("\n2. Pattern Management:")
    success = compiler.add_pattern("artificial intelligence", "AI", "compound", "ai", 950)
    print(f"Pattern added: {success}")
    
    patterns = compiler.get_patterns(domain="ai")
    print(f"AI patterns: {[p['original'] for p in patterns]}")
    
    # 3. System health and statistics
    print("\n3. System Health and Statistics:")
    health = compiler.validate_system_health()
    print(f"System health: {health.overall_status}")
    
    stats = compiler.get_statistics()
    print(f"Total compressions: {stats['session']['compressions']}")
    
    # 4. Engine comparison
    print("\n4. Engine Comparison:")
    comparison = compiler.compare_engines("Implement machine learning pipeline")
    for engine, result in comparison.items():
        if 'error' not in result:
            print(f"{engine}: {result['compression_ratio']:.2f} ratio, {result['quality_score']:.2f} quality")
        else:
            print(f"{engine}: Error - {result['error']}")
    
    # 5. Mock environment setup
    print("\n5. Mock Test Environment:")
    env = MockTestEnvironment()
    test_compiler = env.setup_complete_mock_environment()
    
    test_result = test_compiler.compress("Deploy machine learning model to production")
    print(f"Test compression: {test_result.compressed_text}")
    
    # Generate session report
    print("\n6. Session Report:")
    report = compiler.get_session_report()
    print(report)


class TestPracticalMockImplementations:
    """Test cases demonstrating the mock implementations."""
    
    def test_mock_compiler_basic_functionality(self):
        """Test basic mock compiler functionality."""
        compiler = MockNeuralSemanticCompiler.create_default()
        
        result = compiler.compress("Build a React application")
        
        assert result.original_text == "Build a React application"
        assert result.compressed_text != result.original_text
        assert result.compression_ratio <= 1.0
        assert 0.0 <= result.quality_score <= 1.0
        assert result.original_tokens > 0
        assert result.engine_used in ["SemanticEngine", "HybridEngine", "ExtremeEngine", "FallbackEngine", "SimpleEngine"]
    
    def test_mock_pattern_management(self):
        """Test mock pattern management functionality."""
        compiler = MockNeuralSemanticCompiler()
        
        # Test adding patterns
        assert compiler.add_pattern("test pattern", "TP", "word", "test") is True
        assert compiler.add_pattern("", "invalid", "word", "test") is False  # Invalid input
        
        # Test retrieving patterns
        patterns = compiler.get_patterns(domain="test")
        assert any(p["original"] == "test pattern" for p in patterns)
        
        # Test searching patterns
        search_results = compiler.search_patterns("test")
        assert len(search_results) > 0
    
    def test_mock_vector_store(self):
        """Test mock vector store functionality."""
        vector_store = MockVectorStore(available=True)
        
        # Test adding patterns
        assert vector_store.add_pattern("machine learning", "ML", {"domain": "ai"}) is True
        
        # Test similarity search
        results = vector_store.search_similar("machine learning")
        assert len(results) > 0
        assert results[0]["pattern"] == "machine learning"
        assert results[0]["compressed"] == "ML"
        assert 0.0 <= results[0]["similarity"] <= 1.0
    
    def test_mock_environment_setup(self):
        """Test mock environment setup."""
        env = MockTestEnvironment()
        
        # Test complete environment
        compiler = env.setup_complete_mock_environment()
        assert compiler is not None
        
        result = compiler.compress("Test machine learning implementation")
        assert result.compression_ratio < 1.0
        
        # Test degraded environment
        degraded_compiler = env.setup_degraded_mock_environment()
        health = degraded_compiler.validate_system_health()
        assert health.overall_status in ["warning", "error"]
    
    @pytest.mark.asyncio
    async def test_async_mock_operations(self):
        """Test async mock operations."""
        async_compiler = MockAsyncCompiler()
        
        # Test async compression
        result = await async_compiler.compress_async("Async test")
        assert result.original_text == "Async test"
        
        # Test batch async compression
        texts = ["Text 1", "Text 2", "Text 3"]
        results = await async_compiler.compress_batch_async(texts)
        assert len(results) == len(texts)
        assert all(isinstance(r, MockCompressionResult) for r in results)


if __name__ == "__main__":
    # Run the example usage
    example_usage()
    
    # Run the tests
    print("\n" + "="*60)
    print("Running Mock Implementation Tests...")
    pytest.main([__file__, "-v"])