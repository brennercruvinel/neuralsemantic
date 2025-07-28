"""Main Neural Semantic Compressor implementation."""

import time
import logging
from typing import Dict, Any, Optional
from ..core.types import CompressionResult, CompressionContext, CompressionLevel
from ..patterns import PatternManager, PatternMatcher
from ..vector import VectorStore
from ..compression import ContextAnalyzer, QualityScorer

logger = logging.getLogger(__name__)


class NeuralSemanticCompressor:
    """
    Main Neural Semantic Compressor with multi-stage compression pipeline.
    
    Implements the patent-pending multi-stage compression algorithm with:
    1. Context analysis and strategy selection
    2. Pattern-based compression with conflict resolution
    3. Vector similarity matching for unknown patterns
    4. Quality validation and smart fallback
    5. Real-time learning and optimization
    """

    def __init__(self, pattern_manager: PatternManager, 
                 vector_store: VectorStore = None,
                 context_analyzer: ContextAnalyzer = None,
                 quality_scorer: QualityScorer = None):
        
        self.pattern_manager = pattern_manager
        self.vector_store = vector_store
        self.context_analyzer = context_analyzer or ContextAnalyzer()
        self.quality_scorer = quality_scorer or QualityScorer()
        
        # Performance metrics
        self.compression_stats = {
            'total_compressions': 0,
            'total_processing_time': 0,
            'average_compression_ratio': 0.0,
            'average_quality_score': 0.0,
            'pattern_matches_total': 0
        }

    def compress(self, text: str, **kwargs) -> CompressionResult:
        """
        Main compression method with full pipeline.
        
        Args:
            text: Input text to compress
            **kwargs: Compression parameters (level, domain, etc.)
            
        Returns:
            CompressionResult with compressed text and metrics
        """
        start_time = time.time()
        session_id = kwargs.get('session_id', f"session_{int(time.time())}")
        
        try:
            # Stage 1: Context Analysis
            context = self._analyze_context(text, **kwargs)
            
            # Stage 2: Pattern-based Compression
            compression_result = self._perform_compression(text, context)
            
            # Stage 3: Quality Validation
            quality_result = self._validate_and_optimize_quality(compression_result, context)
            
            # Stage 4: Finalization and Metrics
            final_result = self._finalize_result(quality_result, start_time, session_id)
            
            # Stage 5: Learning Update
            self._update_learning_metrics(final_result, context)
            
            logger.info(f"Compression completed: {len(text)} → {len(final_result.compressed_text)} chars "
                       f"({final_result.compression_ratio:.1%} compression, "
                       f"quality: {final_result.quality_score:.1f}/10)")
            
            return final_result
            
        except Exception as e:
            logger.error(f"Compression failed: {e}")
            # Return safe fallback result
            return self._create_fallback_result(text, start_time, session_id, str(e))

    def _analyze_context(self, text: str, **kwargs) -> CompressionContext:
        """Analyze text context to determine optimal compression strategy."""
        
        # Use context analyzer to get comprehensive analysis
        analysis = self.context_analyzer.analyze(text, **kwargs)
        
        # Create compression context from analysis and kwargs
        context = CompressionContext(
            level=kwargs.get('level', CompressionLevel.BALANCED),
            domain=kwargs.get('domain', analysis.get('domain')),
            language=kwargs.get('language', 'en'),
            preserve_code=kwargs.get('preserve_code', analysis.get('structure_analysis', {}).get('has_code_blocks', False)),
            preserve_urls=kwargs.get('preserve_urls', True),
            preserve_numbers=kwargs.get('preserve_numbers', True),
            target_compression=kwargs.get('target_compression', self._calculate_target_compression(analysis)),
            requires_high_quality=kwargs.get('requires_high_quality', analysis.get('technical_density', 0) > 0.5),
            context_type=analysis.get('content_type', 'general')
        )
        
        logger.debug(f"Context: domain={context.domain}, level={context.level.value}, "
                    f"type={context.context_type}, target={context.target_compression:.1%}")
        
        return context

    def _calculate_target_compression(self, analysis: Dict[str, Any]) -> float:
        """Calculate target compression ratio based on content analysis."""
        base_target = 0.6  # 40% compression
        
        # Adjust based on complexity
        complexity = analysis.get('complexity_score', 0.5)
        if complexity > 0.8:
            base_target += 0.2  # Less aggressive for complex content
        elif complexity < 0.3:
            base_target -= 0.1  # More aggressive for simple content
            
        # Adjust based on technical density
        tech_density = analysis.get('technical_density', 0.0)
        if tech_density > 0.7:
            base_target += 0.15  # Preserve technical content
        elif tech_density < 0.2:
            base_target -= 0.1   # More aggressive for non-technical
            
        # Adjust based on compression readiness
        readiness = analysis.get('compression_readiness', {}).get('overall_score', 0.5)
        if readiness > 0.7:
            base_target -= 0.1  # More aggressive for ready content
        elif readiness < 0.3:
            base_target += 0.1  # Conservative for difficult content
            
        return max(0.2, min(0.9, base_target))

    def _perform_compression(self, text: str, context: CompressionContext) -> CompressionResult:
        """Perform pattern-based compression with conflict resolution."""
        
        # Get relevant patterns
        patterns = self._get_relevant_patterns(context)
        
        if not patterns:
            logger.warning("No patterns available for compression")
            return self._create_minimal_result(text, context)
        
        # Create pattern matcher
        matcher = PatternMatcher(patterns)
        
        # Find pattern matches
        matches = matcher.find_matches(text, context)
        
        if not matches:
            logger.info("No pattern matches found")
            return self._create_minimal_result(text, context)
        
        # Apply compression with fallback protection
        try:
            compressed_text = matcher.apply_matches(text, matches)
            
            # Calculate initial metrics
            compression_ratio = len(compressed_text) / len(text) if text else 1.0
            
            # Create initial result
            result = CompressionResult(
                original_text=text,
                compressed_text=compressed_text,
                original_tokens=self._estimate_tokens(text),
                compressed_tokens=self._estimate_tokens(compressed_text),
                compression_ratio=compression_ratio,
                quality_score=0.0,  # Will be calculated later
                pattern_matches=matches,
                processing_time_ms=0,  # Will be set later
                engine_used="neural_semantic",
                warnings=[]
            )
            
            # Add warnings if compression is too aggressive
            if compression_ratio < 0.3:
                result.warnings.append("Very aggressive compression - quality may be affected")
            elif compression_ratio > 0.9:
                result.warnings.append("Minimal compression achieved")
                
            return result
            
        except Exception as e:
            logger.error(f"Pattern application failed: {e}")
            return self._create_minimal_result(text, context, error=str(e))

    def _get_relevant_patterns(self, context: CompressionContext) -> list:
        """Get patterns relevant to the compression context."""
        
        # Get domain-specific patterns first
        domain_patterns = self.pattern_manager.get_patterns(
            domain=context.domain,
            limit=1000  # Reasonable limit for performance
        )
        
        # Get general patterns as fallback
        general_patterns = self.pattern_manager.get_patterns(
            domain="general",
            limit=500
        )
        
        # Combine patterns with domain patterns having priority
        all_patterns = domain_patterns + [p for p in general_patterns 
                                        if p not in domain_patterns]
        
        # Filter patterns based on context
        filtered_patterns = []
        for pattern in all_patterns:
            if self._is_pattern_suitable(pattern, context):
                filtered_patterns.append(pattern)
                
        logger.debug(f"Using {len(filtered_patterns)} patterns for compression")
        return filtered_patterns

    def _is_pattern_suitable(self, pattern, context: CompressionContext) -> bool:
        """Check if pattern is suitable for the given context."""
        
        # Language compatibility
        if pattern.language != context.language:
            return False
            
        # Code preservation
        if context.preserve_code and self._looks_like_code(pattern.original):
            return False
            
        # URL preservation
        if context.preserve_urls and self._looks_like_url(pattern.original):
            return False
            
        # Domain relevance (allow general patterns always)
        if context.domain and pattern.domain not in [context.domain, "general"]:
            return False
            
        return True

    def _looks_like_code(self, text: str) -> bool:
        """Simple heuristic to detect code patterns."""
        code_indicators = [
            '()', '{}', '[]', '=>', '->', '&&', '||',
            'function', 'def ', 'class ', 'import '
        ]
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in code_indicators)

    def _looks_like_url(self, text: str) -> bool:
        """Simple heuristic to detect URLs."""
        url_indicators = ['http://', 'https://', 'www.', '.com', '.org']
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in url_indicators)

    def _validate_and_optimize_quality(self, result: CompressionResult, 
                                     context: CompressionContext) -> CompressionResult:
        """Validate compression quality and optimize if needed."""
        
        # Calculate comprehensive quality metrics
        quality_metrics = self.quality_scorer.calculate_comprehensive_quality(
            result.original_text, result.compressed_text, result.pattern_matches
        )
        
        # Update result with quality score
        result.quality_score = quality_metrics.composite_score
        result.quality_metrics = quality_metrics
        
        # Check if quality meets minimum threshold
        min_quality = self._get_minimum_quality_threshold(context)
        
        if quality_metrics.composite_score < min_quality:
            logger.warning(f"Quality below threshold: {quality_metrics.composite_score:.1f} < {min_quality}")
            
            # Apply smart recovery strategies
            result = self._apply_quality_recovery(result, context, quality_metrics)
        
        return result

    def _get_minimum_quality_threshold(self, context: CompressionContext) -> float:
        """Get minimum quality threshold based on context."""
        base_threshold = 6.0  # 6/10 minimum quality
        
        # Stricter threshold for high-quality requirements
        if context.requires_high_quality:
            base_threshold = 7.5
            
        # Adjust by compression level
        level_adjustments = {
            CompressionLevel.LIGHT: 1.0,       # Higher threshold
            CompressionLevel.BALANCED: 0.0,    # Base threshold
            CompressionLevel.AGGRESSIVE: -1.0  # Lower threshold
        }
        
        adjusted_threshold = base_threshold + level_adjustments.get(context.level, 0.0)
        return max(4.0, min(9.0, adjusted_threshold))

    def _apply_quality_recovery(self, result: CompressionResult, context: CompressionContext,
                              quality_metrics) -> CompressionResult:
        """Apply quality recovery strategies."""
        
        logger.info("Applying quality recovery strategies")
        
        # Strategy 1: Remove lowest-confidence pattern matches
        if result.pattern_matches:
            # Sort matches by confidence
            sorted_matches = sorted(result.pattern_matches, key=lambda m: m.confidence, reverse=True)
            
            # Try with top 80% of matches
            recovery_matches = sorted_matches[:int(len(sorted_matches) * 0.8)]
            
            if recovery_matches:
                matcher = PatternMatcher([m.pattern for m in recovery_matches])
                recovery_text = matcher.apply_matches(result.original_text, recovery_matches)
                
                # Check if quality improved
                recovery_quality = self.quality_scorer.calculate_quality_score(
                    result.original_text, recovery_text, recovery_matches
                )
                
                if recovery_quality > result.quality_score:
                    logger.info(f"Quality recovery successful: {result.quality_score:.1f} → {recovery_quality:.1f}")
                    
                    # Update result
                    result.compressed_text = recovery_text
                    result.compression_ratio = len(recovery_text) / len(result.original_text)
                    result.compressed_tokens = self._estimate_tokens(recovery_text)
                    result.quality_score = recovery_quality
                    result.pattern_matches = recovery_matches
                    result.warnings.append("Applied quality recovery - removed low-confidence patterns")
        
        # Strategy 2: If still poor quality, use conservative compression
        if result.quality_score < self._get_minimum_quality_threshold(context):
            logger.warning("Falling back to conservative compression")
            result = self._apply_conservative_compression(result, context)
            
        return result

    def _apply_conservative_compression(self, result: CompressionResult, 
                                      context: CompressionContext) -> CompressionResult:
        """Apply conservative compression as fallback."""
        
        # Use only high-priority, high-confidence patterns
        conservative_patterns = [
            pattern for pattern in self.pattern_manager.get_patterns(domain=context.domain)
            if pattern.priority >= 800 and pattern.success_rate >= 0.8
        ]
        
        if conservative_patterns:
            matcher = PatternMatcher(conservative_patterns)
            matches = matcher.find_matches(result.original_text, context)
            
            # Filter to only very confident matches
            confident_matches = [m for m in matches if m.confidence >= 0.9]
            
            if confident_matches:
                conservative_text = matcher.apply_matches(result.original_text, confident_matches)
                conservative_quality = self.quality_scorer.calculate_quality_score(
                    result.original_text, conservative_text, confident_matches
                )
                
                # Update result
                result.compressed_text = conservative_text
                result.compression_ratio = len(conservative_text) / len(result.original_text)
                result.compressed_tokens = self._estimate_tokens(conservative_text)
                result.quality_score = conservative_quality
                result.pattern_matches = confident_matches
                result.warnings.append("Applied conservative compression for quality preservation")
                
                logger.info(f"Conservative compression applied: quality={conservative_quality:.1f}")
        
        return result

    def _finalize_result(self, result: CompressionResult, start_time: float, 
                        session_id: str) -> CompressionResult:
        """Finalize compression result with timing and metadata."""
        
        # Calculate processing time
        processing_time = int((time.time() - start_time) * 1000)
        result.processing_time_ms = processing_time
        result.session_id = session_id
        
        # Update performance statistics
        self.compression_stats['total_compressions'] += 1
        self.compression_stats['total_processing_time'] += processing_time
        self.compression_stats['average_compression_ratio'] = (
            (self.compression_stats['average_compression_ratio'] * 
             (self.compression_stats['total_compressions'] - 1) + 
             result.compression_ratio) / self.compression_stats['total_compressions']
        )
        self.compression_stats['average_quality_score'] = (
            (self.compression_stats['average_quality_score'] * 
             (self.compression_stats['total_compressions'] - 1) + 
             result.quality_score) / self.compression_stats['total_compressions']
        )
        self.compression_stats['pattern_matches_total'] += len(result.pattern_matches)
        
        return result

    def _update_learning_metrics(self, result: CompressionResult, context: CompressionContext):
        """Update learning metrics based on compression result."""
        
        # Update pattern usage statistics
        for match in result.pattern_matches:
            if match.pattern.id:
                # Consider compression successful if quality is good
                success = result.quality_score >= 7.0
                self.pattern_manager.update_usage_stats(match.pattern.id, success)
        
        # Log compression session for analytics
        # This would typically be stored in the database
        logger.debug(f"Session logged: {result.session_id}, "
                    f"quality={result.quality_score:.1f}, "
                    f"ratio={result.compression_ratio:.1%}")

    def _create_minimal_result(self, text: str, context: CompressionContext, 
                             error: str = None) -> CompressionResult:
        """Create minimal compression result when no patterns match."""
        
        # Apply very basic compression (remove extra spaces, etc.)
        compressed = self._apply_basic_compression(text)
        
        return CompressionResult(
            original_text=text,
            compressed_text=compressed,
            original_tokens=self._estimate_tokens(text),
            compressed_tokens=self._estimate_tokens(compressed),
            compression_ratio=len(compressed) / len(text) if text else 1.0,
            quality_score=8.0 if not error else 5.0,  # High quality for minimal compression
            pattern_matches=[],
            processing_time_ms=0,
            engine_used="neural_semantic_minimal",
            warnings=["No pattern matches found - minimal compression applied"] + 
                    ([f"Error: {error}"] if error else [])
        )

    def _apply_basic_compression(self, text: str) -> str:
        """Apply basic compression (whitespace normalization, etc.)."""
        import re
        
        # Normalize whitespace
        compressed = re.sub(r'\s+', ' ', text.strip())
        
        # Remove redundant punctuation
        compressed = re.sub(r'[.]{2,}', '...', compressed)
        compressed = re.sub(r'[!]{2,}', '!', compressed)
        compressed = re.sub(r'[?]{2,}', '?', compressed)
        
        return compressed

    def _create_fallback_result(self, text: str, start_time: float, 
                               session_id: str, error: str) -> CompressionResult:
        """Create fallback result when compression fails completely."""
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return CompressionResult(
            original_text=text,
            compressed_text=text,  # No compression on failure
            original_tokens=self._estimate_tokens(text),
            compressed_tokens=self._estimate_tokens(text),
            compression_ratio=1.0,
            quality_score=5.0,  # Neutral quality
            pattern_matches=[],
            processing_time_ms=processing_time,
            engine_used="neural_semantic_fallback",
            warnings=[f"Compression failed: {error}", "Returned original text"],
            session_id=session_id
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation)."""
        # Simple estimation: ~4 characters per token
        return max(1, len(text) // 4)

    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        stats = self.compression_stats.copy()
        
        if stats['total_compressions'] > 0:
            stats['average_processing_time'] = (
                stats['total_processing_time'] / stats['total_compressions']
            )
            stats['average_patterns_per_compression'] = (
                stats['pattern_matches_total'] / stats['total_compressions']
            )
        else:
            stats['average_processing_time'] = 0
            stats['average_patterns_per_compression'] = 0
            
        return stats

    def reset_performance_stats(self):
        """Reset performance statistics."""
        self.compression_stats = {
            'total_compressions': 0,
            'total_processing_time': 0,
            'average_compression_ratio': 0.0,
            'average_quality_score': 0.0,
            'pattern_matches_total': 0
        }
        logger.info("Performance statistics reset")