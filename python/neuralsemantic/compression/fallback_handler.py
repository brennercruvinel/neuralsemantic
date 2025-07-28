"""Fallback compression strategies for Neural Semantic Compiler."""

import logging
import re
from typing import Dict, Any, List, Optional
from ..core.types import CompressionResult, CompressionContext, CompressionLevel

logger = logging.getLogger(__name__)


class FallbackHandler:
    """
    Handles fallback compression strategies when primary methods fail.
    
    Provides progressive fallback levels:
    1. Conservative pattern matching
    2. Basic text normalization
    3. Whitespace compression only
    4. Return original (no compression)
    """
    
    def __init__(self):
        # Basic compression patterns that are safe to apply
        self.safe_patterns = {
            # Whitespace normalization
            'multiple_spaces': (r'\s+', ' '),
            'trailing_spaces': (r'[ \t]+$', ''),
            'leading_spaces': (r'^[ \t]+', ''),
            
            # Common word shortenings (very conservative)
            'and_symbol': (r'\band\b', '&'),
            'at_symbol': (r'\bat\b', '@'),
            
            # Punctuation normalization
            'multiple_dots': (r'\.{3,}', '...'),
            'multiple_exclamation': (r'!{2,}', '!'),
            'multiple_question': (r'\?{2,}', '?'),
        }
        
        # Safe abbreviations for common phrases
        self.safe_abbreviations = {
            'for example': 'e.g.',
            'that is': 'i.e.',
            'et cetera': 'etc.',
            'versus': 'vs.',
            'with respect to': 'w.r.t.',
            'in other words': 'i.e.',
        }

    def apply_fallback_compression(self, text: str, context: CompressionContext,
                                 fallback_level: int = 1) -> CompressionResult:
        """
        Apply fallback compression strategy.
        
        Args:
            text: Original text to compress
            context: Compression context
            fallback_level: Level of fallback (1-4, higher = more conservative)
            
        Returns:
            CompressionResult with fallback compression applied
        """
        original_text = text
        compressed_text = text
        warnings = []
        
        try:
            if fallback_level == 1:
                # Conservative pattern matching
                compressed_text, applied_patterns = self._apply_conservative_patterns(text, context)
                warnings.append(f"Applied conservative fallback with {len(applied_patterns)} patterns")
                
            elif fallback_level == 2:
                # Basic text normalization
                compressed_text = self._apply_basic_normalization(text)
                warnings.append("Applied basic text normalization fallback")
                
            elif fallback_level == 3:
                # Whitespace compression only
                compressed_text = self._apply_whitespace_compression(text)
                warnings.append("Applied whitespace-only compression fallback")
                
            else:  # fallback_level >= 4
                # No compression - return original
                compressed_text = text
                warnings.append("No compression applied - returned original text")
            
            # Calculate metrics
            compression_ratio = len(compressed_text) / len(original_text) if original_text else 1.0
            quality_score = self._estimate_fallback_quality(original_text, compressed_text, fallback_level)
            
            return CompressionResult(
                original_text=original_text,
                compressed_text=compressed_text,
                original_tokens=self._estimate_tokens(original_text),
                compressed_tokens=self._estimate_tokens(compressed_text),
                compression_ratio=compression_ratio,
                quality_score=quality_score,
                pattern_matches=[],
                processing_time_ms=0,  # Will be set by caller
                engine_used=f"fallback_level_{fallback_level}",
                warnings=warnings
            )
            
        except Exception as e:
            logger.error(f"Fallback compression failed at level {fallback_level}: {e}")
            # Ultimate fallback - return original text
            return self._create_ultimate_fallback(original_text, str(e))

    def _apply_conservative_patterns(self, text: str, context: CompressionContext) -> tuple:
        """Apply only the safest compression patterns."""
        compressed = text
        applied_patterns = []
        
        # Apply safe abbreviations first
        for original, abbrev in self.safe_abbreviations.items():
            if original in text.lower():
                pattern = re.compile(re.escape(original), re.IGNORECASE)
                if pattern.search(compressed):
                    compressed = pattern.sub(abbrev, compressed)
                    applied_patterns.append(f"abbreviation: {original} -> {abbrev}")
        
        # Apply safe regex patterns
        for name, (pattern, replacement) in self.safe_patterns.items():
            before_count = len(compressed)
            compressed = re.sub(pattern, replacement, compressed, flags=re.MULTILINE)
            after_count = len(compressed)
            
            if before_count != after_count:
                applied_patterns.append(f"pattern: {name}")
        
        # Conservative domain-specific patterns
        if context.domain == 'web-development':
            # Safe web dev abbreviations
            web_patterns = {
                'JavaScript': 'JS',
                'TypeScript': 'TS',
                'application programming interface': 'API',
                'user interface': 'UI',
                'user experience': 'UX',
            }
            for original, abbrev in web_patterns.items():
                if original in compressed:
                    compressed = compressed.replace(original, abbrev)
                    applied_patterns.append(f"web: {original} -> {abbrev}")
        
        elif context.domain == 'agile':
            # Safe agile abbreviations
            agile_patterns = {
                'user story': 'story',
                'product owner': 'PO',
                'scrum master': 'SM',
                'definition of done': 'DoD',
            }
            for original, abbrev in agile_patterns.items():
                if original in compressed:
                    compressed = compressed.replace(original, abbrev)
                    applied_patterns.append(f"agile: {original} -> {abbrev}")
        
        return compressed, applied_patterns

    def _apply_basic_normalization(self, text: str) -> str:
        """Apply basic text normalization."""
        normalized = text
        
        # Normalize whitespace
        normalized = re.sub(r'\s+', ' ', normalized)
        normalized = normalized.strip()
        
        # Remove redundant punctuation
        normalized = re.sub(r'\.{3,}', '...', normalized)
        normalized = re.sub(r'!{2,}', '!', normalized)
        normalized = re.sub(r'\?{2,}', '?', normalized)
        
        # Normalize quotes
        normalized = re.sub(r'[""]', '"', normalized)
        normalized = re.sub(r'['']', "'", normalized)
        
        # Remove excessive line breaks
        normalized = re.sub(r'\n{3,}', '\n\n', normalized)
        
        return normalized

    def _apply_whitespace_compression(self, text: str) -> str:
        """Apply only whitespace compression."""
        # Multiple spaces to single space
        compressed = re.sub(r'[ \t]+', ' ', text)
        
        # Multiple newlines to double newlines max
        compressed = re.sub(r'\n{3,}', '\n\n', compressed)
        
        # Remove trailing/leading whitespace on lines
        lines = compressed.split('\n')
        compressed = '\n'.join(line.strip() for line in lines)
        
        # Final trim
        return compressed.strip()

    def _estimate_fallback_quality(self, original: str, compressed: str, level: int) -> float:
        """Estimate quality score for fallback compression."""
        if not original:
            return 5.0
            
        compression_ratio = len(compressed) / len(original)
        
        # Base quality by fallback level
        base_quality = {
            1: 7.0,  # Conservative patterns
            2: 8.0,  # Basic normalization  
            3: 9.0,  # Whitespace only
            4: 10.0  # No compression
        }.get(level, 5.0)
        
        # Adjust for compression achieved
        if compression_ratio < 0.9:  # Some compression achieved
            base_quality += 0.5
        if compression_ratio < 0.8:  # Good compression
            base_quality += 0.5
            
        # Penalty for over-compression at conservative levels
        if level <= 2 and compression_ratio < 0.5:
            base_quality -= 1.0
            
        return max(1.0, min(10.0, base_quality))

    def _create_ultimate_fallback(self, text: str, error: str) -> CompressionResult:
        """Create ultimate fallback result when everything fails."""
        return CompressionResult(
            original_text=text,
            compressed_text=text,  # Return original unchanged
            original_tokens=self._estimate_tokens(text),
            compressed_tokens=self._estimate_tokens(text),
            compression_ratio=1.0,
            quality_score=5.0,  # Neutral quality
            pattern_matches=[],
            processing_time_ms=0,
            engine_used="ultimate_fallback",
            warnings=[
                "All compression methods failed",
                f"Error: {error}",
                "Returned original text unchanged"
            ]
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return max(1, len(text) // 4)

    def suggest_fallback_strategy(self, context: CompressionContext, 
                                error_type: str = None) -> Dict[str, Any]:
        """Suggest appropriate fallback strategy based on context and error."""
        suggestions = {
            'recommended_level': 1,
            'rationale': [],
            'expected_compression': 0.9,
            'expected_quality': 7.0
        }
        
        # Adjust based on compression level
        if context.level == CompressionLevel.LIGHT:
            suggestions['recommended_level'] = 2
            suggestions['rationale'].append("Light compression level - use basic normalization")
            suggestions['expected_compression'] = 0.95
            suggestions['expected_quality'] = 8.5
            
        elif context.level == CompressionLevel.AGGRESSIVE:
            suggestions['recommended_level'] = 1
            suggestions['rationale'].append("Aggressive level - try conservative patterns first")
            suggestions['expected_compression'] = 0.85
            suggestions['expected_quality'] = 7.0
        
        # Adjust based on context requirements
        if context.requires_high_quality:
            suggestions['recommended_level'] = min(3, suggestions['recommended_level'] + 1)
            suggestions['rationale'].append("High quality required - more conservative approach")
            
        # Adjust based on error type
        if error_type == 'pattern_conflict':
            suggestions['recommended_level'] = 2
            suggestions['rationale'].append("Pattern conflicts detected - use basic normalization")
        elif error_type == 'quality_failure':
            suggestions['recommended_level'] = 3
            suggestions['rationale'].append("Quality validation failed - minimal compression only")
        elif error_type == 'timeout':
            suggestions['recommended_level'] = 4
            suggestions['rationale'].append("Processing timeout - skip compression")
            
        return suggestions

    def get_fallback_statistics(self) -> Dict[str, Any]:
        """Get statistics about available fallback methods."""
        return {
            'safe_patterns_count': len(self.safe_patterns),
            'safe_abbreviations_count': len(self.safe_abbreviations),
            'fallback_levels': {
                1: 'Conservative pattern matching',
                2: 'Basic text normalization', 
                3: 'Whitespace compression only',
                4: 'No compression (return original)'
            },
            'estimated_compression_ranges': {
                1: '10-20%',
                2: '5-10%', 
                3: '2-5%',
                4: '0%'
            },
            'quality_preservation': {
                1: 'Good (7-8/10)',
                2: 'Excellent (8-9/10)',
                3: 'Perfect (9-10/10)', 
                4: 'Perfect (10/10)'
            }
        }