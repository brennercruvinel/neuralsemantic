"""Conflict resolution for overlapping patterns."""

import logging
from typing import List, Dict, Set, Tuple, Optional
from ..core.types import Pattern, PatternMatch

logger = logging.getLogger(__name__)


class ConflictResolver:
    """Resolves conflicts between overlapping pattern matches."""

    def __init__(self):
        self.resolution_strategies = {
            'priority': self._resolve_by_priority,
            'length': self._resolve_by_length,
            'confidence': self._resolve_by_confidence,
            'hybrid': self._resolve_hybrid
        }

    def resolve_conflicts(self, matches: List[PatternMatch], 
                         strategy: str = 'hybrid') -> List[PatternMatch]:
        """Resolve conflicts using specified strategy."""
        if not matches:
            return []

        if strategy not in self.resolution_strategies:
            logger.warning(f"Unknown strategy {strategy}, using hybrid")
            strategy = 'hybrid'

        return self.resolution_strategies[strategy](matches)

    def _resolve_by_priority(self, matches: List[PatternMatch]) -> List[PatternMatch]:
        """Resolve conflicts by pattern priority."""
        # Group overlapping matches
        conflict_groups = self._group_overlapping_matches(matches)
        
        resolved = []
        for group in conflict_groups:
            if len(group) == 1:
                resolved.extend(group)
            else:
                # Select highest priority match
                best_match = max(group, key=lambda m: m.pattern.priority)
                resolved.append(best_match)
        
        return sorted(resolved, key=lambda m: m.position)

    def _resolve_by_length(self, matches: List[PatternMatch]) -> List[PatternMatch]:
        """Resolve conflicts by preferring longer matches."""
        conflict_groups = self._group_overlapping_matches(matches)
        
        resolved = []
        for group in conflict_groups:
            if len(group) == 1:
                resolved.extend(group)
            else:
                # Select longest match
                best_match = max(group, key=lambda m: len(m.original_text))
                resolved.append(best_match)
        
        return sorted(resolved, key=lambda m: m.position)

    def _resolve_by_confidence(self, matches: List[PatternMatch]) -> List[PatternMatch]:
        """Resolve conflicts by confidence score."""
        conflict_groups = self._group_overlapping_matches(matches)
        
        resolved = []
        for group in conflict_groups:
            if len(group) == 1:
                resolved.extend(group)
            else:
                # Select highest confidence match
                best_match = max(group, key=lambda m: m.confidence)
                resolved.append(best_match)
        
        return sorted(resolved, key=lambda m: m.position)

    def _resolve_hybrid(self, matches: List[PatternMatch]) -> List[PatternMatch]:
        """Resolve conflicts using hybrid scoring."""
        conflict_groups = self._group_overlapping_matches(matches)
        
        resolved = []
        for group in conflict_groups:
            if len(group) == 1:
                resolved.extend(group)
            else:
                # Calculate hybrid score
                best_match = max(group, key=self._calculate_hybrid_score)
                resolved.append(best_match)
        
        return sorted(resolved, key=lambda m: m.position)

    def _calculate_hybrid_score(self, match: PatternMatch) -> float:
        """Calculate hybrid score for conflict resolution."""
        # Normalize components
        priority_score = min(1.0, match.pattern.priority / 1000)
        length_score = min(1.0, len(match.original_text) / 50)
        confidence_score = match.confidence
        success_score = match.pattern.success_rate
        
        # Weighted combination
        hybrid_score = (
            priority_score * 0.4 +
            confidence_score * 0.3 +
            length_score * 0.2 +
            success_score * 0.1
        )
        
        return hybrid_score

    def _group_overlapping_matches(self, matches: List[PatternMatch]) -> List[List[PatternMatch]]:
        """Group overlapping matches together."""
        if not matches:
            return []
        
        # Sort matches by position
        sorted_matches = sorted(matches, key=lambda m: m.position)
        
        groups = []
        current_group = [sorted_matches[0]]
        
        for i in range(1, len(sorted_matches)):
            current_match = sorted_matches[i]
            last_match = current_group[-1]
            
            # Check if current match overlaps with any match in current group
            if self._matches_overlap(current_match, last_match):
                current_group.append(current_match)
            else:
                # Start new group
                groups.append(current_group)
                current_group = [current_match]
        
        # Add the last group
        groups.append(current_group)
        
        return groups

    def _matches_overlap(self, match1: PatternMatch, match2: PatternMatch) -> bool:
        """Check if two matches overlap."""
        start1 = match1.position
        end1 = start1 + len(match1.original_text)
        
        start2 = match2.position
        end2 = start2 + len(match2.original_text)
        
        return not (end1 <= start2 or end2 <= start1)

    def detect_pattern_conflicts(self, patterns: List[Pattern]) -> List[Dict[str, any]]:
        """Detect potential conflicts between patterns."""
        conflicts = []
        
        for i, pattern1 in enumerate(patterns):
            for j, pattern2 in enumerate(patterns[i+1:], i+1):
                conflict = self._check_pattern_conflict(pattern1, pattern2)
                if conflict:
                    conflicts.append({
                        'pattern1': pattern1,
                        'pattern2': pattern2,
                        'conflict_type': conflict,
                        'severity': self._assess_conflict_severity(pattern1, pattern2, conflict)
                    })
        
        return conflicts

    def _check_pattern_conflict(self, pattern1: Pattern, pattern2: Pattern) -> Optional[str]:
        """Check for conflicts between two patterns."""
        # Circular reference
        if pattern1.original == pattern2.compressed and pattern2.original == pattern1.compressed:
            return 'circular_reference'
        
        # Ambiguous compression (same original, different compressed)
        if pattern1.original == pattern2.original and pattern1.compressed != pattern2.compressed:
            return 'ambiguous_compression'
        
        # Substring conflicts
        if pattern1.original in pattern2.original or pattern2.original in pattern1.original:
            return 'substring_conflict'
        
        # Compressed form conflicts
        if pattern1.compressed == pattern2.compressed and pattern1.original != pattern2.original:
            return 'compressed_collision'
        
        return None

    def _assess_conflict_severity(self, pattern1: Pattern, pattern2: Pattern, 
                                 conflict_type: str) -> str:
        """Assess the severity of a pattern conflict."""
        severity_map = {
            'circular_reference': 'critical',
            'ambiguous_compression': 'high',
            'substring_conflict': 'medium',
            'compressed_collision': 'medium'
        }
        
        base_severity = severity_map.get(conflict_type, 'low')
        
        # Increase severity if patterns have high priority
        if pattern1.priority > 800 or pattern2.priority > 800:
            if base_severity == 'medium':
                return 'high'
            elif base_severity == 'low':
                return 'medium'
        
        return base_severity

    def suggest_resolutions(self, conflict: Dict[str, any]) -> List[str]:
        """Suggest resolutions for a pattern conflict."""
        conflict_type = conflict['conflict_type']
        pattern1 = conflict['pattern1']
        pattern2 = conflict['pattern2']
        
        suggestions = []
        
        if conflict_type == 'circular_reference':
            suggestions.extend([
                "Remove one of the patterns",
                "Modify compression to break the circular reference"
            ])
        
        elif conflict_type == 'ambiguous_compression':
            suggestions.extend([
                f"Use domain-specific variants ({pattern1.domain} vs {pattern2.domain})",
                "Choose the higher priority pattern",
                "Merge patterns with context-aware selection"
            ])
        
        elif conflict_type == 'substring_conflict':
            suggestions.extend([
                "Adjust pattern priorities to prefer longer matches",
                "Use word boundary constraints",
                "Create hierarchical pattern matching"
            ])
        
        elif conflict_type == 'compressed_collision':
            suggestions.extend([
                "Use different compressed forms",
                "Add domain or context suffixes",
                "Merge patterns if semantically similar"
            ])
        
        return suggestions