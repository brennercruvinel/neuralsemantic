"""Advanced similarity search engine with multiple metrics."""

import logging
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from ..core.types import Pattern, SimilarPattern, CompressionContext
from .embeddings import EmbeddingManager

logger = logging.getLogger(__name__)


class SimilaritySearchEngine:
    """Advanced similarity search with multiple metrics and weighting."""

    def __init__(self, embedding_manager: EmbeddingManager):
        self.embedding_manager = embedding_manager
        self.domain_weights = {
            'general': 1.0,
            'web-development': 1.2,
            'agile': 1.1,
            'devops': 1.15
        }

    def find_similar_patterns_advanced(self, text: str, patterns: List[Pattern],
                                     context: CompressionContext,
                                     n_results: int = 10) -> List[SimilarPattern]:
        """Multi-dimensional similarity search with semantic validation."""
        if not patterns:
            return []

        # Generate context-aware embeddings
        query_embedding = self.embedding_manager.get_embedding(text)
        
        # Filter patterns by basic criteria
        candidate_patterns = self._filter_candidates(patterns, context)
        
        if not candidate_patterns:
            return []

        # Calculate similarities
        scored_patterns = []
        for pattern in candidate_patterns:
            similarity_metrics = self._calculate_multi_similarity(
                text, pattern, query_embedding, context
            )
            
            if similarity_metrics['composite_score'] >= self._get_dynamic_threshold(text, context):
                similar_pattern = SimilarPattern(
                    original=pattern.original,
                    compressed=pattern.compressed,
                    similarity=similarity_metrics['composite_score'],
                    pattern_type=pattern.pattern_type.value,
                    domain=pattern.domain,
                    priority=pattern.priority,
                    confidence=self._calculate_confidence(similarity_metrics, context),
                    semantic_breakdown=similarity_metrics
                )
                scored_patterns.append(similar_pattern)

        # Rank and return top matches
        scored_patterns.sort(key=lambda x: x.similarity, reverse=True)
        return scored_patterns[:n_results]

    def _filter_candidates(self, patterns: List[Pattern], 
                          context: CompressionContext) -> List[Pattern]:
        """Filter patterns by basic compatibility criteria."""
        filtered = []
        
        for pattern in patterns:
            # Language check
            if pattern.language != context.language:
                continue
                
            # Domain compatibility
            if (context.domain and 
                pattern.domain not in ['general', context.domain]):
                continue
                
            # Basic suitability checks
            if self._is_pattern_contextually_suitable(pattern, context):
                filtered.append(pattern)
        
        return filtered

    def _is_pattern_contextually_suitable(self, pattern: Pattern, 
                                        context: CompressionContext) -> bool:
        """Check contextual suitability of pattern."""
        # Preserve code blocks
        if context.preserve_code and self._looks_like_code(pattern.original):
            return False
            
        # Preserve URLs
        if context.preserve_urls and self._looks_like_url(pattern.original):
            return False
            
        # Preserve numbers in certain contexts
        if context.preserve_numbers and pattern.original.isdigit():
            return False
            
        return True

    def _calculate_multi_similarity(self, text: str, pattern: Pattern,
                                  query_embedding: np.ndarray,
                                  context: CompressionContext) -> Dict[str, float]:
        """Calculate multiple similarity metrics."""
        # Semantic similarity using embeddings
        pattern_embedding = self.embedding_manager.get_embedding(pattern.original)
        cosine_sim = self._cosine_similarity(query_embedding, pattern_embedding)
        
        # Structural similarity (for technical terms)
        structural_sim = self._calculate_structural_similarity(text, pattern.original)
        
        # Contextual similarity (domain relevance)
        context_sim = self._calculate_contextual_similarity(pattern, context)
        
        # Lexical similarity (character/word overlap)
        lexical_sim = self._calculate_lexical_similarity(text, pattern.original)
        
        # Compression efficiency potential
        compression_potential = self._calculate_compression_potential(pattern)
        
        # Composite similarity score with dynamic weighting
        weights = self._get_similarity_weights(context)
        composite_score = (
            cosine_sim * weights['semantic'] +
            structural_sim * weights['structural'] +
            context_sim * weights['contextual'] +
            lexical_sim * weights['lexical'] +
            compression_potential * weights['compression']
        )
        
        return {
            'cosine': cosine_sim,
            'structural': structural_sim,
            'contextual': context_sim,
            'lexical': lexical_sim,
            'compression_potential': compression_potential,
            'composite_score': composite_score
        }

    def _cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between embeddings."""
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
            
        return max(0.0, dot_product / (norm1 * norm2))

    def _calculate_structural_similarity(self, text1: str, text2: str) -> float:
        """Calculate structural similarity for technical terms."""
        # Normalize texts
        norm1 = text1.lower().strip()
        norm2 = text2.lower().strip()
        
        # Exact match
        if norm1 == norm2:
            return 1.0
            
        # Substring matching with position weighting
        if norm2 in norm1:
            # Weight by position (earlier = better)
            pos = norm1.find(norm2)
            position_weight = 1.0 - (pos / len(norm1))
            length_weight = len(norm2) / len(norm1)
            return 0.7 + 0.3 * (position_weight * length_weight)
        
        if norm1 in norm2:
            pos = norm2.find(norm1)
            position_weight = 1.0 - (pos / len(norm2))
            length_weight = len(norm1) / len(norm2)
            return 0.7 + 0.3 * (position_weight * length_weight)
        
        # Word overlap
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1 & words2
        union = words1 | words2
        
        jaccard = len(intersection) / len(union)
        return jaccard * 0.8  # Reduce weight for partial matches

    def _calculate_contextual_similarity(self, pattern: Pattern, 
                                       context: CompressionContext) -> float:
        """Calculate contextual relevance score."""
        base_score = 0.5
        
        # Domain matching bonus
        if context.domain:
            if pattern.domain == context.domain:
                base_score += 0.4
            elif pattern.domain == 'general':
                base_score += 0.2
            else:
                base_score -= 0.2
        
        # Priority weighting
        priority_boost = min(0.3, pattern.priority / 3333)  # Max 0.3 boost
        
        # Success rate bonus
        success_boost = pattern.success_rate * 0.2
        
        return min(1.0, base_score + priority_boost + success_boost)

    def _calculate_lexical_similarity(self, text1: str, text2: str) -> float:
        """Calculate lexical overlap similarity."""
        # Character-level similarity for short texts
        if len(text1) <= 10 or len(text2) <= 10:
            return self._character_similarity(text1, text2)
        
        # Word-level similarity for longer texts
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
            
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)

    def _character_similarity(self, text1: str, text2: str) -> float:
        """Calculate character-level similarity using edit distance."""
        s1, s2 = text1.lower(), text2.lower()
        
        # Levenshtein distance
        m, n = len(s1), len(s2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # Initialize
        for i in range(m + 1):
            dp[i][0] = i
        for j in range(n + 1):
            dp[0][j] = j
            
        # Fill DP table
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if s1[i-1] == s2[j-1]:
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = 1 + min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1])
        
        # Convert to similarity (0-1)
        max_len = max(m, n)
        if max_len == 0:
            return 1.0
            
        return 1.0 - (dp[m][n] / max_len)

    def _calculate_compression_potential(self, pattern: Pattern) -> float:
        """Calculate compression potential score."""
        original_len = len(pattern.original)
        compressed_len = len(pattern.compressed)
        
        if original_len == 0:
            return 0.0
            
        # Compression ratio (higher is better)
        compression_ratio = 1.0 - (compressed_len / original_len)
        
        # Bonus for significant compression
        if compression_ratio > 0.5:
            return min(1.0, compression_ratio + 0.2)
        
        return compression_ratio

    def _get_similarity_weights(self, context: CompressionContext) -> Dict[str, float]:
        """Get dynamic similarity weights based on context."""
        if context.level.value == 'aggressive':
            return {
                'semantic': 0.3,
                'structural': 0.2,
                'contextual': 0.2,
                'lexical': 0.1,
                'compression': 0.2
            }
        elif context.level.value == 'light':
            return {
                'semantic': 0.5,
                'structural': 0.2,
                'contextual': 0.2,
                'lexical': 0.1,
                'compression': 0.0
            }
        else:  # balanced
            return {
                'semantic': 0.4,
                'structural': 0.2,
                'contextual': 0.2,
                'lexical': 0.1,
                'compression': 0.1
            }

    def _get_dynamic_threshold(self, text: str, context: CompressionContext) -> float:
        """Calculate dynamic threshold based on text and context characteristics."""
        base_threshold = 0.6
        
        # Adjust for text length
        text_length = len(text.split())
        if text_length > 20:
            base_threshold -= 0.1  # More lenient for longer text
        elif text_length < 5:
            base_threshold += 0.1  # More strict for short text
        
        # Adjust for compression level
        level_adjustments = {
            'light': 0.2,
            'balanced': 0.0,
            'aggressive': -0.15
        }
        base_threshold += level_adjustments.get(context.level.value, 0.0)
        
        # Adjust for domain specificity
        if context.domain and context.domain != 'general':
            base_threshold -= 0.05  # Slightly more lenient for domain-specific
        
        return max(0.3, min(0.9, base_threshold))

    def _calculate_confidence(self, similarity_metrics: Dict[str, float], 
                            context: CompressionContext) -> float:
        """Calculate confidence score for the match."""
        composite = similarity_metrics['composite_score']
        
        # Base confidence from composite score
        confidence = composite * 0.8
        
        # Boost for high semantic similarity
        if similarity_metrics['cosine'] > 0.8:
            confidence += 0.1
            
        # Boost for structural matches
        if similarity_metrics['structural'] > 0.9:
            confidence += 0.1
            
        # Context consistency bonus
        if similarity_metrics['contextual'] > 0.8:
            confidence += 0.05
        
        return min(1.0, confidence)

    def _looks_like_code(self, text: str) -> bool:
        """Detect code-like patterns."""
        code_indicators = [
            '()', '{}', '[]', '=>', '->', '&&', '||',
            'function', 'def ', 'class ', 'import ',
            'return ', 'if ', 'for ', 'while ', '===', '!=='
        ]
        
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in code_indicators)

    def _looks_like_url(self, text: str) -> bool:
        """Detect URL patterns."""
        url_indicators = ['http://', 'https://', 'www.', '.com', '.org', '.net', '://']
        text_lower = text.lower()
        return any(indicator in text_lower for indicator in url_indicators)

    def explain_similarity(self, text: str, pattern: Pattern, 
                          context: CompressionContext) -> Dict[str, Any]:
        """Provide detailed explanation of similarity calculation."""
        query_embedding = self.embedding_manager.get_embedding(text)
        metrics = self._calculate_multi_similarity(text, pattern, query_embedding, context)
        weights = self._get_similarity_weights(context)
        
        return {
            'query_text': text,
            'pattern_original': pattern.original,
            'pattern_compressed': pattern.compressed,
            'similarity_metrics': metrics,
            'weights_used': weights,
            'threshold': self._get_dynamic_threshold(text, context),
            'explanation': {
                'semantic': f"Embedding similarity: {metrics['cosine']:.3f}",
                'structural': f"Text structure match: {metrics['structural']:.3f}",
                'contextual': f"Domain/context relevance: {metrics['contextual']:.3f}",
                'lexical': f"Word/character overlap: {metrics['lexical']:.3f}",
                'compression': f"Compression potential: {metrics['compression_potential']:.3f}",
                'final_score': f"Weighted composite: {metrics['composite_score']:.3f}"
            }
        }