"""Base compression engine interface."""

import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from ..core.types import CompressionResult, CompressionContext, PatternMatch
from ..core.exceptions import CompressionError

logger = logging.getLogger(__name__)


class BaseCompressionEngine(ABC):
    """Abstract base class for compression engines."""

    def __init__(self, pattern_manager, vector_store=None, config=None):
        self.pattern_manager = pattern_manager
        self.vector_store = vector_store
        self.config = config or {}
        self.engine_name = self.__class__.__name__.replace('CompressionEngine', '').lower()

    @abstractmethod
    def compress(self, text: str, context: CompressionContext) -> CompressionResult:
        """Compress text using this engine's strategy."""
        pass

    def _create_compression_result(self, original_text: str, compressed_text: str,
                                 pattern_matches: List[PatternMatch],
                                 processing_time_ms: int,
                                 warnings: List[str] = None) -> CompressionResult:
        """Create a compression result with token counting."""
        
        original_tokens = self._estimate_token_count(original_text)
        compressed_tokens = self._estimate_token_count(compressed_text)
        
        compression_ratio = len(compressed_text) / len(original_text) if original_text else 1.0
        
        return CompressionResult(
            original_text=original_text,
            compressed_text=compressed_text,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            compression_ratio=compression_ratio,
            quality_score=0.0,  # Will be calculated later
            pattern_matches=pattern_matches,
            processing_time_ms=processing_time_ms,
            engine_used=self.engine_name,
            warnings=warnings or []
        )

    def _estimate_token_count(self, text: str) -> int:
        """Estimate token count (simplified version)."""
        if not text:
            return 0
        return max(1, len(text) // 4)

    def _validate_input(self, text: str) -> None:
        """Validate input text."""
        if not text or not text.strip():
            raise CompressionError("Input text is empty or whitespace only")
        
        if len(text) > 100000:  # 100K character limit
            raise CompressionError("Input text too long (max 100K characters)")

    def _should_preserve_segment(self, segment: str, context: CompressionContext) -> bool:
        """Check if a text segment should be preserved from compression."""
        segment_lower = segment.lower()
        
        if context.preserve_code:
            code_indicators = [
                'function', 'def ', 'class ', 'import ', 'from ',
                'return ', 'if ', 'else', 'for ', 'while ',
                '()', '{}', '[]', '=>', '->', '&&', '||'
            ]
            if any(indicator in segment_lower for indicator in code_indicators):
                return True
        
        if context.preserve_urls:
            if 'http://' in segment_lower or 'https://' in segment_lower or 'www.' in segment_lower:
                return True
        
        if context.preserve_numbers:
            import re
            if re.search(r'\b\d+\.\d+\b|\b\d{4,}\b', segment):
                return True
        
        return False

    def _split_into_segments(self, text: str) -> List[Dict[str, Any]]:
        """Split text into segments for selective compression."""
        segments = []
        current_segment = ""
        in_code_block = False
        
        lines = text.split('\n')
        
        for line in lines:
            line_lower = line.strip().lower()
            
            if '```' in line or line_lower.startswith('```'):
                in_code_block = not in_code_block
                if current_segment:
                    segments.append({
                        'text': current_segment,
                        'type': 'code' if in_code_block else 'text',
                        'compressible': not in_code_block
                    })
                    current_segment = ""
                continue
            
            current_segment += line + '\n'
            
            if not line.strip() and current_segment.strip():
                segments.append({
                    'text': current_segment.rstrip('\n'),
                    'type': 'code' if in_code_block else 'text',
                    'compressible': not in_code_block
                })
                current_segment = ""
        
        if current_segment.strip():
            segments.append({
                'text': current_segment.rstrip('\n'),
                'type': 'code' if in_code_block else 'text',
                'compressible': not in_code_block
            })
        
        return segments

    def _log_compression_stats(self, result: CompressionResult) -> None:
        """Log compression statistics."""
        char_reduction = len(result.original_text) - len(result.compressed_text)
        token_reduction = result.original_tokens - result.compressed_tokens
        
        logger.info(
            f"{self.engine_name} compression: "
            f"{char_reduction} chars saved ({result.compression_ratio:.1%}), "
            f"{token_reduction} tokens saved, "
            f"{len(result.pattern_matches)} patterns applied, "
            f"{result.processing_time_ms}ms"
        )
