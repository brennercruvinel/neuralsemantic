"""Semantic compression engine with quality preservation."""

import time
import logging
from typing import List, Dict, Any

from .base_engine import BaseCompressionEngine
from ..core.types import CompressionResult, CompressionContext, PatternMatch
from ..patterns.pattern_matcher import PatternMatcher
from ..core.exceptions import CompressionError

logger = logging.getLogger(__name__)


class SemanticCompressionEngine(BaseCompressionEngine):
    """
    Semantic compression engine focused on quality preservation.
    Uses conservative approach with semantic validation.
    """

    def __init__(self, pattern_manager, vector_store=None, config=None):
        super().__init__(pattern_manager, vector_store, config)
        self.quality_threshold = config.compression.semantic_threshold if config else 0.90

    def compress(self, text: str, context: CompressionContext) -> CompressionResult:
        """
        Compress text using semantic approach with quality validation.
        
        Multi-stage pipeline:
        1. Input validation and segmentation
        2. High-priority pattern matching
        3. Semantic similarity enhancement  
        4. Quality validation with rollback
        5. Result compilation
        """
        start_time = time.time()
        
        try:
            self._validate_input(text)
            
            segments = self._split_into_segments(text)
            all_matches = []
            compressed_segments = []
            warnings = []
            
            for segment in segments:
                if not segment['compressible']:
                    compressed_segments.append(segment['text'])
                    continue
                
                patterns = self._get_semantic_patterns(context)
                matcher = PatternMatcher(patterns)
                
                matches = matcher.find_matches(segment['text'], context)
                
                validated_matches = self._validate_semantic_matches(
                    segment['text'], matches, context
                )
                
                compressed_segment = self._apply_matches_with_validation(
                    segment['text'], validated_matches, context
                )
                
                compressed_segments.append(compressed_segment)
                all_matches.extend(validated_matches)
                
                if len(validated_matches) < len(matches):
                    warnings.append(f"Filtered {len(matches) - len(validated_matches)} low-quality matches")
            
            final_compressed = '\n'.join(compressed_segments)
            
            if self._calculate_semantic_similarity(text, final_compressed) < self.quality_threshold:
                warnings.append("Quality threshold not met, using conservative compression")
                final_compressed = self._apply_conservative_compression(text, context)
            
            processing_time = int((time.time() - start_time) * 1000)
            
            result = self._create_compression_result(
                text, final_compressed, all_matches, processing_time, warnings
            )
            
            self._log_compression_stats(result)
            return result
            
        except Exception as e:
            logger.error(f"Semantic compression failed: {e}")
            raise CompressionError(f"Semantic compression failed: {e}") from e

    def _get_semantic_patterns(self, context: CompressionContext) -> List:
        """Get patterns optimized for semantic compression."""
        # Prioritize high-quality, well-tested patterns
        patterns = self.pattern_manager.get_patterns(
            domain=context.domain,
            pattern_type=None
        )
        
        # Filter for semantic engine preferences
        semantic_patterns = []
        for pattern in patterns:
            # Prefer patterns with high success rate
            if pattern.success_rate >= 0.8:
                semantic_patterns.append(pattern)
            # Include high-priority patterns even with lower success rate
            elif pattern.priority >= 800:
                semantic_patterns.append(pattern)
        
        # Sort by quality metrics
        semantic_patterns.sort(
            key=lambda p: (p.success_rate * 0.6 + (p.priority / 1000) * 0.4),
            reverse=True
        )
        
        return semantic_patterns

    def _validate_semantic_matches(self, text: str, matches: List[PatternMatch],
                                 context: CompressionContext) -> List[PatternMatch]:
        """Validate matches using semantic similarity."""
        validated_matches = []
        
        for match in matches:
            # Create text with this match applied
            test_text = text.replace(match.original_text, match.compressed_text, 1)
            
            # Calculate semantic similarity
            similarity = self._calculate_semantic_similarity(text, test_text)
            
            # Check if similarity meets threshold
            if similarity >= self.quality_threshold:
                validated_matches.append(match)
            else:
                logger.debug(f"Rejected match: '{match.original_text}' -> '{match.compressed_text}' "
                           f"(similarity: {similarity:.3f})")
        
        return validated_matches

    def _apply_matches_with_validation(self, text: str, matches: List[PatternMatch],
                                     context: CompressionContext) -> str:
        """Apply matches with incremental validation."""
        current_text = text
        applied_matches = []
        
        # Sort matches by confidence and apply incrementally
        sorted_matches = sorted(matches, key=lambda m: m.confidence, reverse=True)
        
        for match in sorted_matches:
            # Apply this match
            test_text = current_text.replace(match.original_text, match.compressed_text, 1)
            
            # Validate quality
            if self._calculate_semantic_similarity(text, test_text) >= self.quality_threshold:
                current_text = test_text
                applied_matches.append(match)
            else:
                logger.debug(f"Incremental validation failed for: {match.original_text}")
        
        return current_text

    def _apply_conservative_compression(self, text: str, context: CompressionContext) -> str:
        """Apply only the most conservative, high-confidence patterns."""
        # Get only highest priority patterns
        conservative_patterns = self.pattern_manager.get_patterns(
            domain=context.domain
        )
        
        # Filter for only the most reliable patterns
        reliable_patterns = [
            p for p in conservative_patterns 
            if p.priority >= 900 and p.success_rate >= 0.95
        ]
        
        if not reliable_patterns:
            return text  # No safe compression available
        
        matcher = PatternMatcher(reliable_patterns)
        matches = matcher.find_matches(text, context)
        
        # Apply only high-confidence matches
        high_confidence_matches = [m for m in matches if m.confidence >= 0.9]
        
        return matcher.apply_matches(text, high_confidence_matches)

    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts."""
        if not self.vector_store or not self.vector_store.enabled:
            # Fallback to simple similarity
            return self._simple_text_similarity(text1, text2)
        
        try:
            # Use embedding-based similarity
            return self.vector_store.embedding_manager.similarity(text1, text2)
        except Exception as e:
            logger.warning(f"Vector similarity failed, using fallback: {e}")
            return self._simple_text_similarity(text1, text2)

    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity fallback."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        # Prevent division by zero
        if not union:
            return 0.0
            
        return len(intersection) / len(union)

    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        # Simple implementation - in production use NLP libraries
        words = text.lower().split()
        
        # Filter out common stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
            'before', 'after', 'above', 'below', 'between', 'among', 'is', 'are',
            'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does',
            'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can'
        }
        
        key_concepts = []
        for word in words:
            # Clean word
            clean_word = ''.join(c for c in word if c.isalnum())
            
            # Include if significant
            if (len(clean_word) > 3 and 
                clean_word not in stop_words and 
                not clean_word.isdigit()):
                key_concepts.append(clean_word)
        
        return key_concepts

    def _calculate_concept_preservation(self, original: str, compressed: str) -> float:
        """Calculate how well key concepts are preserved."""
        original_concepts = set(self._extract_key_concepts(original))
        compressed_concepts = set(self._extract_key_concepts(compressed))
        
        if not original_concepts:
            return 1.0
        
        preserved = original_concepts.intersection(compressed_concepts)
        preservation_ratio = len(preserved) / len(original_concepts)
        
        return preservation_ratio
