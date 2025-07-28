"""Hybrid compression engine balancing quality and compression."""

import time
import logging
import re
from typing import List, Dict, Any

from .base_engine import BaseCompressionEngine
from .semantic_engine import SemanticCompressionEngine
from .extreme_engine import ExtremeCompressionEngine
from ..core.types import CompressionResult, CompressionContext, PatternMatch
from ..core.exceptions import CompressionError

logger = logging.getLogger(__name__)


class HybridCompressionEngine(BaseCompressionEngine):
    """
    Hybrid compression engine that balances quality and compression ratio.
    Dynamically chooses strategies based on context and content.
    """

    def __init__(self, pattern_manager, vector_store=None, config=None):
        super().__init__(pattern_manager, vector_store, config)
        
        # Initialize sub-engines
        self.semantic_engine = SemanticCompressionEngine(pattern_manager, vector_store, config)
        self.extreme_engine = ExtremeCompressionEngine(pattern_manager, vector_store, config)
        
        # Hybrid configuration
        # Quality threshold: Minimum semantic similarity score required (0.85 = 85% similarity)
        self.quality_threshold = config.compression.semantic_threshold if config else 0.85
        # Target compression ratio: 0.6 means reducing text to 60% of original size (40% reduction)
        self.target_compression = 0.6

    def compress(self, text: str, context: CompressionContext) -> CompressionResult:
        """
        Compress text using hybrid approach.
        
        Adaptive pipeline:
        1. Content analysis and strategy selection
        2. Multi-engine processing with quality monitoring
        3. Adaptive refinement based on results
        4. Quality-compression optimization
        """
        start_time = time.time()
        
        try:
            # Stage 1: Analyze content and select strategy
            self._validate_input(text)
            content_profile = self._analyze_content(text, context)
            strategy = self._select_strategy(content_profile, context)
            
            # Stage 2: Apply hybrid compression
            result = self._apply_hybrid_compression(text, context, strategy, content_profile)
            
            # Stage 3: Adaptive refinement if needed
            if self._needs_refinement(result, context):
                result = self._refine_compression(result, context, content_profile)
            
            processing_time = int((time.time() - start_time) * 1000)
            result.processing_time_ms = processing_time
            
            self._log_compression_stats(result)
            return result
            
        except Exception as e:
            logger.error(f"Hybrid compression failed: {e}")
            raise CompressionError(f"Hybrid compression failed: {e}") from e

    def _analyze_content(self, text: str, context: CompressionContext) -> Dict[str, Any]:
        """Analyze content characteristics to guide compression strategy."""
        profile = {
            'length': len(text),
            'word_count': len(text.split()),
            'technical_density': self._calculate_technical_density(text),
            'structural_complexity': self._calculate_structural_complexity(text),
            'domain_specificity': self._calculate_domain_specificity(text, context),
            'compression_potential': self._estimate_compression_potential(text),
            'quality_sensitivity': self._estimate_quality_sensitivity(text, context)
        }
        
        return profile

    def _calculate_technical_density(self, text: str) -> float:
        """Calculate how technical/specialized the text is."""
        technical_indicators = [
            'api', 'database', 'server', 'client', 'function', 'method', 'class',
            'algorithm', 'implementation', 'architecture', 'framework', 'library',
            'interface', 'protocol', 'authentication', 'authorization', 'configuration'
        ]
        
        words = text.lower().split()
        technical_count = sum(1 for word in words if any(indicator in word for indicator in technical_indicators))
        
        return technical_count / len(words) if words else 0.0

    def _calculate_structural_complexity(self, text: str) -> float:
        """Calculate structural complexity of the text."""
        # Count various structural elements
        sentences = text.count('.') + text.count('!') + text.count('?')
        commas = text.count(',')
        semicolons = text.count(';')
        parentheses = text.count('(') + text.count(')')
        lists = text.count('-') + text.count('*') + text.count('•')
        
        words = len(text.split())
        if words == 0:
            return 0.0
        
        # Normalize by word count
        complexity = (sentences + commas * 0.5 + semicolons * 0.8 + parentheses * 0.3 + lists * 0.6) / words
        
        return min(1.0, complexity * 10)  # Scale to 0-1

    def _calculate_domain_specificity(self, text: str, context: CompressionContext) -> float:
        """Calculate how domain-specific the text is."""
        if not context.domain or context.domain == 'general':
            return 0.5
        
        # Domain-specific keywords
        domain_keywords = {
            'web-development': ['react', 'javascript', 'html', 'css', 'api', 'frontend', 'backend', 'database'],
            'agile': ['sprint', 'scrum', 'backlog', 'story', 'epic', 'retrospective', 'standup', 'velocity'],
            'devops': ['docker', 'kubernetes', 'ci/cd', 'pipeline', 'deployment', 'monitoring', 'infrastructure']
        }
        
        keywords = domain_keywords.get(context.domain, [])
        if not keywords:
            return 0.5
        
        text_lower = text.lower()
        keyword_count = sum(1 for keyword in keywords if keyword in text_lower)
        
        return min(1.0, keyword_count / len(keywords))

    def _estimate_compression_potential(self, text: str) -> float:
        """Estimate how much the text can be compressed."""
        # Factors that indicate high compression potential
        long_words = sum(1 for word in text.split() if len(word) > 8)
        repeated_phrases = self._count_repeated_phrases(text)
        verbose_patterns = self._count_verbose_patterns(text)
        redundancy = self._estimate_redundancy(text)
        
        total_words = len(text.split())
        if total_words == 0:
            return 0.0
        
        # Normalize factors
        potential = (
            (long_words / total_words) * 0.3 +
            (repeated_phrases / total_words) * 0.3 +
            (verbose_patterns / total_words) * 0.2 +
            redundancy * 0.2
        )
        
        return min(1.0, potential)

    def _count_repeated_phrases(self, text: str) -> int:
        """Count repeated phrases in text."""
        words = text.lower().split()
        phrase_counts = {}
        repeated_count = 0
        
        # Check 2-4 word phrases
        for phrase_len in [2, 3, 4]:
            for i in range(len(words) - phrase_len + 1):
                phrase = ' '.join(words[i:i + phrase_len])
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
                
                if phrase_counts[phrase] == 2:  # First repetition
                    repeated_count += phrase_len
        
        return repeated_count

    def _count_verbose_patterns(self, text: str) -> int:
        """Count verbose patterns that can be simplified."""
        verbose_patterns = [
            'in order to', 'due to the fact that', 'for the purpose of',
            'with respect to', 'in terms of', 'as a result of',
            'in spite of the fact that', 'regardless of the fact that'
        ]
        
        text_lower = text.lower()
        count = sum(text_lower.count(pattern) for pattern in verbose_patterns)
        
        return count

    def _estimate_redundancy(self, text: str) -> float:
        """Estimate text redundancy."""
        words = text.lower().split()
        unique_words = set(words)
        
        if not words:
            return 0.0
        
        # Simple redundancy measure
        redundancy = 1.0 - (len(unique_words) / len(words))
        
        return redundancy

    def _estimate_quality_sensitivity(self, text: str, context: CompressionContext) -> float:
        """Estimate how sensitive the text is to quality loss."""
        # Factors that indicate high quality sensitivity
        
        # Technical documentation is quality-sensitive
        technical_sensitivity = self._calculate_technical_density(text) * 0.4
        
        # Complex structures are quality-sensitive
        structural_sensitivity = self._calculate_structural_complexity(text) * 0.3
        
        # Formal language is quality-sensitive
        formal_indicators = ['therefore', 'however', 'furthermore', 'consequently', 'nevertheless']
        formal_count = sum(1 for indicator in formal_indicators if indicator in text.lower())
        formal_sensitivity = min(0.3, formal_count / len(text.split()) * 10) if text.split() else 0.0
        
        total_sensitivity = technical_sensitivity + structural_sensitivity + formal_sensitivity
        
        return min(1.0, total_sensitivity)

    def _select_strategy(self, content_profile: Dict[str, Any], context: CompressionContext) -> str:
        """Select compression strategy based on content analysis."""
        
        # Factor weights for strategy selection
        quality_sensitivity = content_profile['quality_sensitivity']
        compression_potential = content_profile['compression_potential']
        technical_density = content_profile['technical_density']
        
        # Strategy selection logic
        if quality_sensitivity > 0.7:
            # High quality sensitivity - use semantic approach
            return 'semantic'
        elif compression_potential > 0.8 and quality_sensitivity < 0.4:
            # High compression potential, low quality sensitivity - use extreme
            return 'extreme'
        elif context.level.value == 'aggressive':
            # User explicitly wants aggressive compression
            return 'extreme'
        elif context.level.value == 'light':
            # User wants conservative compression
            return 'semantic'
        else:
            # Default to hybrid approach
            return 'hybrid'

    def _apply_hybrid_compression(self, text: str, context: CompressionContext,
                                strategy: str, content_profile: Dict[str, Any]) -> CompressionResult:
        """Apply compression using selected strategy."""
        
        if strategy == 'semantic':
            return self.semantic_engine.compress(text, context)
        elif strategy == 'extreme':
            return self.extreme_engine.compress(text, context)
        else:  # hybrid
            return self._apply_true_hybrid(text, context, content_profile)

    def _apply_true_hybrid(self, text: str, context: CompressionContext,
                          content_profile: Dict[str, Any]) -> CompressionResult:
        """Apply true hybrid compression combining both approaches."""
        
        # Split text into segments with different strategies
        segments = self._split_into_segments(text)
        compressed_segments = []
        all_matches = []
        warnings = []
        
        for segment in segments:
            if not segment['compressible']:
                compressed_segments.append(segment['text'])
                continue
            
            # Analyze segment
            segment_profile = self._analyze_content(segment['text'], context)
            
            # Choose strategy for this segment
            if segment_profile['quality_sensitivity'] > 0.6:
                # Use semantic for quality-sensitive segments
                try:
                    result = self.semantic_engine.compress(segment['text'], context)
                    compressed_segments.append(result.compressed_text)
                    all_matches.extend(result.pattern_matches)
                except Exception:
                    # Fallback to original
                    compressed_segments.append(segment['text'])
                    warnings.append("Semantic compression failed for segment")
            else:
                # Use extreme for other segments
                try:
                    result = self.extreme_engine.compress(segment['text'], context)
                    compressed_segments.append(result.compressed_text)
                    all_matches.extend(result.pattern_matches)
                except Exception:
                    # Fallback to original
                    compressed_segments.append(segment['text'])
                    warnings.append("Extreme compression failed for segment")
        
        # Combine segments
        final_compressed = '\n'.join(compressed_segments)
        
        # Apply final optimizations
        final_compressed = self._apply_hybrid_optimizations(final_compressed, context)
        
        return self._create_compression_result(
            text, final_compressed, all_matches, 0, warnings
        )

    def _apply_hybrid_optimizations(self, text: str, context: CompressionContext) -> str:
        """Apply final hybrid optimizations."""
        # Clean up transitions between segments
        optimized = text
        
        # Remove redundant line breaks
        optimized = re.sub(r'\n\s*\n\s*\n', '\n\n', optimized)
        
        # Clean up spaces
        optimized = re.sub(r' +', ' ', optimized)
        
        # Remove trailing spaces
        lines = optimized.split('\n')
        lines = [line.rstrip() for line in lines]
        optimized = '\n'.join(lines)
        
        return optimized

    def _needs_refinement(self, result: CompressionResult, context: CompressionContext) -> bool:
        """Check if compression result needs refinement."""
        
        # Check if compression ratio is too low
        if result.compression_ratio > 0.8 and context.target_compression < 0.7:
            return True
        
        # Check if compression ratio is too high (potential quality loss)
        if result.compression_ratio < 0.3:
            return True
        
        # Check if there are quality warnings
        if any('quality' in warning.lower() for warning in result.warnings):
            return True
        
        return False

    def _refine_compression(self, result: CompressionResult, context: CompressionContext,
                          content_profile: Dict[str, Any]) -> CompressionResult:
        """Refine compression result based on analysis."""
        
        if result.compression_ratio > 0.8:
            # Compression too low - try more aggressive approach
            try:
                refined_result = self.extreme_engine.compress(result.original_text, context)
                if refined_result.compression_ratio < result.compression_ratio:
                    result.warnings.append("Applied aggressive refinement")
                    return refined_result
            except Exception:
                pass
        
        elif result.compression_ratio < 0.3:
            # Compression too high - try more conservative approach
            try:
                refined_result = self.semantic_engine.compress(result.original_text, context)
                result.warnings.append("Applied conservative refinement")
                return refined_result
            except Exception:
                pass
        
        return result
