"""Core type definitions for Neural Semantic Compiler."""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional
import time


class CompressionLevel(Enum):
    """Compression level enum."""
    NONE = "none"           # Code blocks, no compression
    LIGHT = "light"         # Headers, minimal compression  
    BALANCED = "balanced"   # Paragraphs, moderate compression
    AGGRESSIVE = "aggressive" # Lists, maximum compression


class PatternType(Enum):
    """Pattern type enum."""
    PHRASE = "phrase"
    COMPOUND = "compound" 
    WORD = "word"
    ABBREVIATION = "abbreviation"
    STRUCTURE = "structure"


@dataclass
class Pattern:
    """Compression pattern definition."""
    id: Optional[int] = None
    original: str = ""
    compressed: str = ""
    pattern_type: PatternType = PatternType.WORD
    priority: int = 500
    domain: str = "general"
    language: str = "en"
    frequency: int = 0
    success_rate: float = 0.0
    version: int = 1
    is_active: bool = True
    metadata: Optional[Dict[str, Any]] = None
    created_at: Optional[float] = field(default_factory=time.time)
    updated_at: Optional[float] = field(default_factory=time.time)

    @classmethod
    def from_row(cls, row: Any) -> "Pattern":
        """Create Pattern from database row."""
        # Handle both sqlite3.Row and dict objects
        def safe_get(key, default=None):
            try:
                return row[key] if row[key] is not None else default
            except (KeyError, IndexError):
                return default
        
        return cls(
            id=safe_get('id'),
            original=safe_get('original', ''),
            compressed=safe_get('compressed', ''),
            pattern_type=PatternType(safe_get('pattern_type', 'word')),
            priority=safe_get('priority', 500),
            domain=safe_get('domain', 'general'),
            language=safe_get('language', 'en'),
            frequency=safe_get('frequency', 0),
            success_rate=safe_get('success_rate', 0.0),
            version=safe_get('version', 1),
            is_active=bool(safe_get('is_active', True)),
            metadata=safe_get('metadata'),
            created_at=safe_get('created_at'),
            updated_at=safe_get('updated_at')
        )


@dataclass 
class PatternMatch:
    """Pattern match result."""
    pattern: Pattern
    position: int
    original_text: str
    compressed_text: str
    confidence: float
    context: str = ""


@dataclass
class CompressionContext:
    """Context for compression operation."""
    level: CompressionLevel = CompressionLevel.BALANCED
    domain: Optional[str] = None
    language: str = "en"
    preserve_code: bool = True
    preserve_urls: bool = True
    preserve_numbers: bool = True
    target_compression: float = 0.6
    requires_high_quality: bool = True
    context_type: str = "general"


@dataclass
class QualityMetrics:
    """Quality assessment metrics."""
    composite_score: float
    semantic_preservation: float
    information_density: float
    compression_efficiency: float
    llm_interpretability: float
    structural_preservation: float
    linguistic_coherence: float
    entity_preservation: float
    breakdown_details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompressionResult:
    """Result of compression operation."""
    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    quality_score: float
    pattern_matches: List[PatternMatch]
    processing_time_ms: int
    engine_used: str
    warnings: List[str] = field(default_factory=list)
    quality_metrics: Optional[QualityMetrics] = None
    session_id: Optional[str] = None

    @property
    def token_savings(self) -> int:
        """Calculate token savings."""
        return self.original_tokens - self.compressed_tokens

    @property  
    def savings_percentage(self) -> float:
        """Calculate savings percentage."""
        return (1 - self.compression_ratio) * 100


@dataclass
class SimilarPattern:
    """Similar pattern from vector search."""
    original: str
    compressed: str
    similarity: float
    pattern_type: str
    domain: str
    priority: int
    confidence: float = 0.0
    semantic_breakdown: Dict[str, float] = field(default_factory=dict)


@dataclass
class NewPattern:
    """New discovered pattern."""
    original: str
    compressed: str
    frequency: int
    estimated_savings: float
    confidence: float


@dataclass
class CandidatePhrase:
    """Candidate phrase for pattern learning."""
    text: str
    frequency: int
    length: int
    word_count: int


@dataclass
class SemanticConcept:
    """Semantic concept for analysis."""
    text: str
    type: str
    importance: float
    context_relevance: float