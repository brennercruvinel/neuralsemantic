"""Pattern matching engine for compression."""

import re
import logging
from typing import List, Dict, Set, Tuple
from ..core.types import Pattern, PatternMatch, CompressionContext

logger = logging.getLogger(__name__)


class PatternMatcher:
    """Pattern matching engine with conflict resolution and optimization."""

    def __init__(self, patterns: List[Pattern]):
        self.patterns = patterns
        self._build_pattern_index()

    def _build_pattern_index(self) -> None:
        """Build efficient pattern index for fast matching."""
        self.pattern_index: Dict[str, List[Pattern]] = {}
        self.sorted_patterns = sorted(self.patterns, key=lambda p: (-p.priority, -len(p.original)))
        
        # Index by first word for faster lookup
        for pattern in self.sorted_patterns:
            first_word = pattern.original.split()[0].lower()
            if first_word not in self.pattern_index:
                self.pattern_index[first_word] = []
            self.pattern_index[first_word].append(pattern)

    def find_matches(self, text: str, context: CompressionContext) -> List[PatternMatch]:
        """Find all pattern matches in text with conflict resolution."""
        matches = []
        text_lower = text.lower()
        
        # Find all potential matches
        potential_matches = self._find_potential_matches(text, text_lower, context)
        
        # Resolve conflicts and optimize
        final_matches = self._resolve_conflicts(potential_matches, text)
        
        return final_matches

    def _find_potential_matches(self, text: str, text_lower: str, 
                              context: CompressionContext) -> List[PatternMatch]:
        """Find all potential pattern matches."""
        potential_matches = []
        
        for pattern in self.sorted_patterns:
            # Skip patterns not suitable for context
            if not self._is_pattern_suitable(pattern, context):
                continue
                
            # Find all occurrences of this pattern
            pattern_lower = pattern.original.lower()
            start = 0
            
            while True:
                pos = text_lower.find(pattern_lower, start)
                if pos == -1:
                    break
                
                # Check word boundaries for whole word patterns
                if self._is_valid_match(text, pos, pattern):
                    match = PatternMatch(
                        pattern=pattern,
                        position=pos,
                        original_text=text[pos:pos + len(pattern.original)],
                        compressed_text=pattern.compressed,
                        confidence=self._calculate_confidence(pattern, text, pos),
                        context=context.context_type
                    )
                    potential_matches.append(match)
                
                start = pos + 1
        
        return potential_matches

    def _is_pattern_suitable(self, pattern: Pattern, context: CompressionContext) -> bool:
        """Check if pattern is suitable for given context."""
        # Domain matching
        if context.domain and pattern.domain != "general" and pattern.domain != context.domain:
            return False
        
        # Language matching
        if pattern.language != context.language:
            return False
        
        # Context-specific rules
        if context.preserve_code and self._looks_like_code(pattern.original):
            return False
            
        return True

    def _is_valid_match(self, text: str, pos: int, pattern: Pattern) -> bool:
        """Check if match is valid (word boundaries, etc.)."""
        # Check word boundaries for word-type patterns
        if pattern.pattern_type.value == "word":
            # Check start boundary
            if pos > 0 and text[pos - 1].isalnum():
                return False
            
            # Check end boundary
            end_pos = pos + len(pattern.original)
            if end_pos < len(text) and text[end_pos].isalnum():
                return False
        
        return True

    def _calculate_confidence(self, pattern: Pattern, text: str, pos: int) -> float:
        """Calculate match confidence score."""
        base_confidence = 0.8
        
        # Boost for higher priority patterns
        priority_boost = min(0.2, pattern.priority / 5000)
        
        # Boost for higher success rate
        success_boost = pattern.success_rate * 0.1
        
        # Context boost (simple heuristics)
        context_boost = 0.0
        surrounding = text[max(0, pos-20):pos+len(pattern.original)+20].lower()
        
        # Technical context detection
        tech_keywords = ['api', 'database', 'server', 'client', 'function', 'code']
        if any(keyword in surrounding for keyword in tech_keywords):
            if pattern.domain in ['web-development', 'general']:
                context_boost = 0.1
        
        return min(1.0, base_confidence + priority_boost + success_boost + context_boost)

    def _resolve_conflicts(self, potential_matches: List[PatternMatch], 
                          text: str) -> List[PatternMatch]:
        """Resolve overlapping matches and optimize selection."""
        if not potential_matches:
            return []
        
        # Sort by position and priority
        potential_matches.sort(key=lambda m: (m.position, -m.pattern.priority))
        
        final_matches = []
        used_positions: Set[int] = set()
        
        for match in potential_matches:
            match_range = set(range(match.position, match.position + len(match.original_text)))
            
            # Check for overlap
            if not match_range.intersection(used_positions):
                final_matches.append(match)
                used_positions.update(match_range)
            else:
                # Check if this match is better than existing overlapping matches
                overlapping_matches = [
                    m for m in final_matches 
                    if set(range(m.position, m.position + len(m.original_text))).intersection(match_range)
                ]
                
                if overlapping_matches:
                    best_existing = max(overlapping_matches, key=lambda m: m.pattern.priority)
                    
                    # Replace if new match is significantly better
                    if (match.pattern.priority > best_existing.pattern.priority * 1.2 or
                        match.confidence > best_existing.confidence + 0.2):
                        
                        # Remove overlapping matches
                        for old_match in overlapping_matches:
                            final_matches.remove(old_match)
                            old_range = set(range(old_match.position, 
                                                old_match.position + len(old_match.original_text)))
                            used_positions -= old_range
                        
                        # Add new match
                        final_matches.append(match)
                        used_positions.update(match_range)
        
        # Sort final matches by position
        final_matches.sort(key=lambda m: m.position)
        return final_matches

    def _looks_like_code(self, text: str) -> bool:
        """Simple heuristic to detect code-like patterns."""
        code_indicators = [
            '()', '{}', '[]', '=>', '->', '&&', '||', '++', '--',
            'function', 'def ', 'class ', 'import ', 'from ',
            'return ', 'if ', 'else', 'for ', 'while '
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in code_indicators)

    def apply_matches(self, text: str, matches: List[PatternMatch]) -> str:
        """Apply pattern matches to compress text."""
        if not matches:
            return text
        
        # Sort matches by position in reverse order to avoid position shifts
        matches.sort(key=lambda m: m.position, reverse=True)
        
        result = text
        for match in matches:
            start = match.position
            end = start + len(match.original_text)
            result = result[:start] + match.compressed_text + result[end:]
        
        return result

    def get_compression_stats(self, text: str, matches: List[PatternMatch]) -> Dict[str, any]:
        """Get compression statistics for matches."""
        if not matches:
            return {
                'total_matches': 0,
                'char_reduction': 0,
                'estimated_token_reduction': 0,
                'patterns_by_type': {}
            }
        
        total_char_reduction = sum(
            len(match.original_text) - len(match.compressed_text) 
            for match in matches
        )
        
        patterns_by_type = {}
        for match in matches:
            pattern_type = match.pattern.pattern_type.value
            if pattern_type not in patterns_by_type:
                patterns_by_type[pattern_type] = 0
            patterns_by_type[pattern_type] += 1
        
        # Rough estimation: 1 token ≈ 4 characters
        estimated_token_reduction = total_char_reduction // 4
        
        return {
            'total_matches': len(matches),
            'char_reduction': total_char_reduction,
            'estimated_token_reduction': estimated_token_reduction,
            'patterns_by_type': patterns_by_type,
            'avg_confidence': sum(m.confidence for m in matches) / len(matches)
        }