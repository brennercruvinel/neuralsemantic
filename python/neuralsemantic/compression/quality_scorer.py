"""Quality scoring system for Neural Semantic Compiler."""

import re
import logging
from typing import Dict, Any, List, Optional
import math

from ..core.types import CompressionContext, PatternMatch, QualityMetrics

logger = logging.getLogger(__name__)


class QualityScorer:
    """
    Multi-dimensional quality assessment for semantic compression.
    
    Evaluates compression quality across multiple dimensions:
    - Semantic preservation
    - Information density  
    - Compression efficiency
    - LLM interpretability
    - Structural preservation
    """

    def __init__(self):
        self.weights = {
            'semantic_preservation': 0.35,
            'information_density': 0.20,
            'compression_efficiency': 0.15,
            'llm_interpretability': 0.15,
            'structural_preservation': 0.10,
            'linguistic_coherence': 0.03,
            'entity_preservation': 0.02
        }
        
        # Common technical terms that should be preserved
        self.critical_terms = {
            'technical': ['api', 'database', 'server', 'client', 'function', 'variable'],
            'agile': ['sprint', 'scrum', 'backlog', 'story', 'retrospective'],
            'web': ['react', 'component', 'interface', 'frontend', 'backend']
        }

    def calculate_comprehensive_quality(self, original: str, compressed: str,
                                      context: CompressionContext) -> QualityMetrics:
        """Calculate comprehensive quality metrics."""
        try:
            # Calculate individual dimension scores
            semantic_preservation = self._assess_semantic_preservation(original, compressed)
            information_density = self._calculate_information_density(original, compressed)
            compression_efficiency = self._calculate_compression_efficiency(original, compressed)
            llm_interpretability = self._assess_llm_interpretability(compressed, context)
            structural_preservation = self._assess_structural_preservation(original, compressed)
            linguistic_coherence = self._assess_linguistic_coherence(compressed)
            entity_preservation = self._assess_entity_preservation(original, compressed)
            
            # Calculate weighted composite score
            composite_score = (
                semantic_preservation * self.weights['semantic_preservation'] +
                information_density * self.weights['information_density'] +
                compression_efficiency * self.weights['compression_efficiency'] +
                llm_interpretability * self.weights['llm_interpretability'] +
                structural_preservation * self.weights['structural_preservation'] +
                linguistic_coherence * self.weights['linguistic_coherence'] +
                entity_preservation * self.weights['entity_preservation']
            ) * 10  # Scale to 0-10
            
            # Calculate breakdown details
            breakdown_details = {
                'compression_ratio': len(compressed) / len(original),
                'character_reduction': len(original) - len(compressed),
                'word_count_ratio': len(compressed.split()) / len(original.split()) if original.split() else 1.0,
                'preserved_critical_terms': self._count_preserved_critical_terms(original, compressed, context),
                'semantic_similarity_estimate': semantic_preservation
            }
            
            return QualityMetrics(
                composite_score=composite_score,
                semantic_preservation=semantic_preservation,
                information_density=information_density,
                compression_efficiency=compression_efficiency,
                llm_interpretability=llm_interpretability,
                structural_preservation=structural_preservation,
                linguistic_coherence=linguistic_coherence,
                entity_preservation=entity_preservation,
                breakdown_details=breakdown_details
            )
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            # Return minimal quality metrics on error
            return QualityMetrics(
                composite_score=5.0,  # Neutral score
                semantic_preservation=0.5,
                information_density=0.5,
                compression_efficiency=0.5,
                llm_interpretability=0.5,
                structural_preservation=0.5,
                linguistic_coherence=0.5,
                entity_preservation=0.5,
                breakdown_details={'error': str(e)}
            )

    def _assess_semantic_preservation(self, original: str, compressed: str) -> float:
        """Assess semantic similarity between original and compressed text."""
        # Basic lexical similarity
        original_words = set(original.lower().split())
        compressed_words = set(compressed.lower().split())
        
        if not original_words:
            return 1.0
        
        # Jaccard similarity
        intersection = len(original_words & compressed_words)
        union = len(original_words | compressed_words)
        jaccard_similarity = intersection / union if union > 0 else 0
        
        # Key concept preservation
        key_concepts = self._extract_key_concepts(original)
        preserved_concepts = sum(1 for concept in key_concepts if concept.lower() in compressed.lower())
        concept_preservation = preserved_concepts / len(key_concepts) if key_concepts else 1.0
        
        # Structure preservation (approximate)
        structure_score = self._assess_sentence_structure_similarity(original, compressed)
        
        # Weighted combination
        semantic_score = (
            jaccard_similarity * 0.4 +
            concept_preservation * 0.4 +
            structure_score * 0.2
        )
        
        return min(1.0, semantic_score)

    def _calculate_information_density(self, original: str, compressed: str) -> float:
        """Calculate information density (information per character)."""
        if not compressed:
            return 0.0
            
        # Extract key information units
        original_concepts = set(self._extract_key_concepts(original))
        compressed_concepts = set(self._extract_key_concepts(compressed))
        
        # Information preservation ratio
        preserved_info = len(original_concepts & compressed_concepts)
        total_info = len(original_concepts)
        info_ratio = preserved_info / total_info if total_info > 0 else 1.0
        
        # Character efficiency
        char_efficiency = len(compressed) / len(original) if len(original) > 0 else 1.0
        
        # Information density = preserved info / character reduction
        if char_efficiency < 1.0:
            density = info_ratio / char_efficiency
        else:
            density = info_ratio
            
        return min(1.0, density)

    def _calculate_compression_efficiency(self, original: str, compressed: str) -> float:
        """Calculate compression efficiency score."""
        if not original:
            return 1.0
            
        compression_ratio = len(compressed) / len(original)
        
        # Optimal compression range is 0.3-0.7
        if 0.3 <= compression_ratio <= 0.7:
            # Within optimal range
            efficiency = 1.0 - abs(compression_ratio - 0.5) * 2
        elif compression_ratio < 0.3:
            # Too aggressive, penalize
            efficiency = 0.5 - (0.3 - compression_ratio)
        else:
            # Not enough compression
            efficiency = 1.0 - (compression_ratio - 0.7) * 2
            
        return max(0.0, min(1.0, efficiency))

    def _assess_llm_interpretability(self, compressed: str, context: CompressionContext) -> float:
        """Assess how interpretable the compressed text is for LLMs."""
        score = 0.8  # Base score
        
        # Check for common abbreviation patterns
        abbreviation_ratio = len(re.findall(r'\b[A-Z]{2,}\b', compressed)) / len(compressed.split())
        if abbreviation_ratio > 0.3:  # Too many abbreviations
            score -= 0.2
        elif abbreviation_ratio > 0.5:
            score -= 0.4
            
        # Check for preserved context clues
        context_clues = ['with', 'for', 'in', 'on', 'at', 'to', 'from']
        preserved_clues = sum(1 for clue in context_clues if clue in compressed.lower())
        clue_score = preserved_clues / len(context_clues)
        score += clue_score * 0.1
        
        # Check for domain-specific term preservation
        if context.domain and context.domain in self.critical_terms:
            domain_terms = self.critical_terms[context.domain]
            preserved_terms = sum(1 for term in domain_terms if term in compressed.lower())
            domain_score = preserved_terms / len(domain_terms)
            score += domain_score * 0.1
        
        # Penalty for excessive symbol usage
        symbol_ratio = len(re.findall(r'[^\w\s]', compressed)) / len(compressed)
        if symbol_ratio > 0.2:
            score -= (symbol_ratio - 0.2) * 0.5
            
        return max(0.0, min(1.0, score))

    def _assess_structural_preservation(self, original: str, compressed: str) -> float:
        """Assess preservation of text structure."""
        # Sentence count preservation
        original_sentences = len([s for s in original.split('.') if s.strip()])
        compressed_sentences = len([s for s in compressed.split('.') if s.strip()])
        
        if original_sentences > 0:
            sentence_ratio = min(1.0, compressed_sentences / original_sentences)
        else:
            sentence_ratio = 1.0
            
        # Paragraph structure (rough approximation)
        original_paragraphs = len([p for p in original.split('\n\n') if p.strip()])
        compressed_paragraphs = len([p for p in compressed.split('\n\n') if p.strip()])
        
        if original_paragraphs > 1:
            paragraph_ratio = min(1.0, compressed_paragraphs / original_paragraphs)
        else:
            paragraph_ratio = 1.0
            
        # List structure preservation
        original_lists = len(re.findall(r'^\s*[-*•]\s', original, re.MULTILINE))
        compressed_lists = len(re.findall(r'^\s*[-*•]\s', compressed, re.MULTILINE))
        
        if original_lists > 0:
            list_ratio = min(1.0, compressed_lists / original_lists)
        else:
            list_ratio = 1.0
            
        # Weighted structure score
        structure_score = (
            sentence_ratio * 0.5 +
            paragraph_ratio * 0.3 +
            list_ratio * 0.2
        )
        
        return structure_score

    def _assess_linguistic_coherence(self, compressed: str) -> float:
        """Assess linguistic coherence of compressed text."""
        if not compressed.strip():
            return 0.0
            
        words = compressed.split()
        if len(words) < 2:
            return 0.5
            
        coherence_score = 0.8  # Base score
        
        # Check for broken words or excessive abbreviations
        broken_words = sum(1 for word in words if len(word) == 1 and word.isalpha())
        if broken_words > len(words) * 0.3:
            coherence_score -= 0.3
            
        # Check for proper spacing
        spacing_issues = len(re.findall(r'\w[^\w\s]\w', compressed))
        if spacing_issues > 0:
            coherence_score -= min(0.2, spacing_issues * 0.05)
            
        # Check for readability indicators
        avg_word_length = sum(len(word) for word in words) / len(words)
        if avg_word_length < 2:  # Too many single-character "words"
            coherence_score -= 0.2
        elif avg_word_length > 12:  # Words too long
            coherence_score -= 0.1
            
        return max(0.0, min(1.0, coherence_score))

    def _assess_entity_preservation(self, original: str, compressed: str) -> float:
        """Assess preservation of named entities and important terms."""
        # Extract potential entities (capitalized words, numbers, etc.)
        original_entities = set(re.findall(r'\b[A-Z][a-z]+\b|\b\d+\b', original))
        compressed_entities = set(re.findall(r'\b[A-Z][a-z]+\b|\b\d+\b', compressed))
        
        if not original_entities:
            return 1.0
            
        preserved_entities = len(original_entities & compressed_entities)
        preservation_ratio = preserved_entities / len(original_entities)
        
        return preservation_ratio

    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key concepts from text."""
        words = text.lower().split()
        
        # Filter out stop words and short words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should'}
        
        key_concepts = [
            word for word in words 
            if len(word) > 3 and word not in stop_words
        ]
        
        # Add multi-word concepts
        bigrams = [f"{words[i]} {words[i+1]}" for i in range(len(words)-1)]
        technical_bigrams = [
            bigram for bigram in bigrams 
            if any(tech_word in bigram for tech_word in ['api', 'web', 'data', 'user', 'system'])
        ]
        
        return key_concepts + technical_bigrams

    def _assess_sentence_structure_similarity(self, original: str, compressed: str) -> float:
        """Assess similarity in sentence structure."""
        original_sentences = [s.strip() for s in original.split('.') if s.strip()]
        compressed_sentences = [s.strip() for s in compressed.split('.') if s.strip()]
        
        if not original_sentences:
            return 1.0
            
        # Simple structure assessment based on sentence count and average length
        length_ratio = len(compressed_sentences) / len(original_sentences)
        
        if len(original_sentences) > 0 and len(compressed_sentences) > 0:
            orig_avg_length = sum(len(s.split()) for s in original_sentences) / len(original_sentences)
            comp_avg_length = sum(len(s.split()) for s in compressed_sentences) / len(compressed_sentences)
            
            if orig_avg_length > 0:
                avg_length_ratio = comp_avg_length / orig_avg_length
            else:
                avg_length_ratio = 1.0
        else:
            avg_length_ratio = 1.0
            
        # Structure similarity combines both ratios
        structure_similarity = (length_ratio + avg_length_ratio) / 2
        
        # Clamp to reasonable range
        return max(0.2, min(1.0, structure_similarity))

    def _count_preserved_critical_terms(self, original: str, compressed: str, 
                                       context: CompressionContext) -> Dict[str, Any]:
        """Count critical terms preserved during compression."""
        original_lower = original.lower()
        compressed_lower = compressed.lower()
        
        results = {}
        
        # Check domain-specific terms
        if context.domain and context.domain in self.critical_terms:
            domain_terms = self.critical_terms[context.domain]
            original_count = sum(1 for term in domain_terms if term in original_lower)
            preserved_count = sum(1 for term in domain_terms if term in compressed_lower)
            
            results['domain_terms'] = {
                'original': original_count,
                'preserved': preserved_count,
                'preservation_rate': preserved_count / original_count if original_count > 0 else 1.0
            }
        
        # Check general technical terms
        all_technical = [term for terms in self.critical_terms.values() for term in terms]
        original_tech_count = sum(1 for term in all_technical if term in original_lower)
        preserved_tech_count = sum(1 for term in all_technical if term in compressed_lower)
        
        results['technical_terms'] = {
            'original': original_tech_count,
            'preserved': preserved_tech_count,
            'preservation_rate': preserved_tech_count / original_tech_count if original_tech_count > 0 else 1.0
        }
        
        return results

    def calculate_quality_score(self, original: str, compressed: str, 
                               pattern_matches: List[PatternMatch]) -> float:
        """Calculate simplified quality score for backward compatibility."""
        if not original or not compressed:
            return 0.0
            
        # Basic quality assessment
        base_score = 8.0
        
        # Compression ratio assessment
        compression_ratio = len(compressed) / len(original)
        if compression_ratio < 0.3:
            base_score -= 2.0
        elif compression_ratio < 0.5:
            base_score -= 1.0
        elif compression_ratio > 0.9:
            base_score -= 1.5
            
        # Pattern match quality boost
        if pattern_matches:
            avg_confidence = sum(m.confidence for m in pattern_matches) / len(pattern_matches)
            base_score += avg_confidence * 1.5
            
        # Word preservation check
        original_words = set(original.lower().split())
        compressed_words = set(compressed.lower().split())
        if original_words:
            word_preservation = len(original_words & compressed_words) / len(original_words)
            if word_preservation < 0.3:
                base_score -= 2.0
            elif word_preservation < 0.5:
                base_score -= 1.0
                
        return max(0.0, min(10.0, base_score))