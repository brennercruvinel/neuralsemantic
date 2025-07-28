"""Extreme compression engine for maximum token reduction."""

import time
import logging
import re
from typing import List, Dict, Any

from .base_engine import BaseCompressionEngine
from ..core.types import CompressionResult, CompressionContext, PatternMatch
from ..patterns.pattern_matcher import PatternMatcher
from ..core.exceptions import CompressionError

logger = logging.getLogger(__name__)


class ExtremeCompressionEngine(BaseCompressionEngine):
    """
    Extreme compression engine for maximum token reduction.
    Trades some quality for aggressive compression ratios.
    """

    def __init__(self, pattern_manager, vector_store=None, config=None):
        super().__init__(pattern_manager, vector_store, config)
        self.min_quality_threshold = 0.75  # Lower threshold for extreme compression

    def compress(self, text: str, context: CompressionContext) -> CompressionResult:
        """
        Compress text using extreme approach for maximum token reduction.
        
        Aggressive pipeline:
        1. Input validation and preprocessing
        2. Multi-stage pattern application
        3. Aggressive abbreviation
        4. Syntactic reduction
        5. Post-processing optimization
        """
        start_time = time.time()
        
        try:
            # Stage 1: Validation and preprocessing
            self._validate_input(text)
            
            segments = self._split_into_segments(text)
            all_matches = []
            compressed_segments = []
            warnings = []
            
            # Stage 2: Aggressive compression per segment
            for segment in segments:
                if not segment['compressible']:
                    compressed_segments.append(segment['text'])
                    continue
                
                # Apply multi-stage extreme compression
                compressed_segment, segment_matches = self._apply_extreme_compression(
                    segment['text'], context
                )
                
                compressed_segments.append(compressed_segment)
                all_matches.extend(segment_matches)
            
            # Stage 3: Global post-processing
            final_compressed = '\n'.join(compressed_segments)
            final_compressed = self._apply_global_optimizations(final_compressed, context)
            
            # Quality check with relaxed threshold
            quality = self._estimate_quality(text, final_compressed)
            if quality < self.min_quality_threshold:
                warnings.append(f"Quality below threshold: {quality:.2f}")
            
            processing_time = int((time.time() - start_time) * 1000)
            
            result = self._create_compression_result(
                text, final_compressed, all_matches, processing_time, warnings
            )
            
            self._log_compression_stats(result)
            return result
            
        except Exception as e:
            logger.error(f"Extreme compression failed: {e}")
            raise CompressionError(f"Extreme compression failed: {e}") from e

    def _apply_extreme_compression(self, text: str, context: CompressionContext) -> tuple:
        """Apply multi-stage extreme compression."""
        current_text = text
        all_matches = []
        
        # Stage 1: High-priority patterns (900-1000)
        patterns_high = self._get_patterns_by_priority(context, min_priority=900)
        if patterns_high:
            matcher = PatternMatcher(patterns_high)
            matches = matcher.find_matches(current_text, context)
            current_text = matcher.apply_matches(current_text, matches)
            all_matches.extend(matches)
        
        # Stage 2: Medium-priority patterns (700-899)
        patterns_medium = self._get_patterns_by_priority(context, min_priority=700, max_priority=899)
        if patterns_medium:
            matcher = PatternMatcher(patterns_medium)
            matches = matcher.find_matches(current_text, context)
            current_text = matcher.apply_matches(current_text, matches)
            all_matches.extend(matches)
        
        # Stage 3: Low-priority patterns (500-699)
        patterns_low = self._get_patterns_by_priority(context, min_priority=500, max_priority=699)
        if patterns_low:
            matcher = PatternMatcher(patterns_low)
            matches = matcher.find_matches(current_text, context)
            current_text = matcher.apply_matches(current_text, matches)
            all_matches.extend(matches)
        
        # Stage 4: Aggressive abbreviation
        current_text = self._apply_aggressive_abbreviation(current_text, context)
        
        # Stage 5: Syntactic reduction
        current_text = self._apply_syntactic_reduction(current_text, context)
        
        return current_text, all_matches

    def _get_patterns_by_priority(self, context: CompressionContext, 
                                min_priority: int, max_priority: int = None) -> List:
        """Get patterns within priority range."""
        all_patterns = self.pattern_manager.get_patterns(domain=context.domain)
        
        filtered_patterns = []
        for pattern in all_patterns:
            if pattern.priority >= min_priority:
                if max_priority is None or pattern.priority <= max_priority:
                    filtered_patterns.append(pattern)
        
        return filtered_patterns

    def _apply_aggressive_abbreviation(self, text: str, context: CompressionContext) -> str:
        """Apply aggressive abbreviation strategies."""
        compressed = text
        
        # Remove articles aggressively
        compressed = self._remove_articles(compressed)
        
        # Compress common phrases
        compressed = self._compress_common_phrases(compressed)
        
        # Apply vowel compression for technical terms
        compressed = self._apply_vowel_compression(compressed)
        
        # Compress repeated words
        compressed = self._compress_repetitions(compressed)
        
        return compressed

    def _remove_articles(self, text: str) -> str:
        """Remove articles and prepositions where safe."""
        # Define removable words
        removable = {
            'the ', 'a ', 'an ', 'and ', 'or ', 'but ', 'in ', 'on ', 'at ',
            'to ', 'for ', 'of ', 'with ', 'by ', 'from ', 'is ', 'are ',
            'was ', 'were ', 'will ', 'would ', 'should ', 'could '
        }
        
        compressed = text
        words = compressed.split()
        
        # Only remove if it doesn't break meaning
        result_words = []
        for i, word in enumerate(words):
            word_lower = word.lower() + ' '
            
            # Check if safe to remove
            if word_lower in removable:
                # Keep if it's at sentence start or essential for meaning
                if i == 0 or self._is_essential_word(word, words, i):
                    result_words.append(word)
                # Otherwise skip it
            else:
                result_words.append(word)
        
        return ' '.join(result_words)

    def _is_essential_word(self, word: str, words: List[str], index: int) -> bool:
        """Check if a word is essential for meaning."""
        word_lower = word.lower()
        
        # Keep "is/are" if followed by important words
        if word_lower in ['is', 'are']:
            if index + 1 < len(words):
                next_word = words[index + 1].lower()
                if next_word in ['not', 'required', 'needed', 'important']:
                    return True
        
        # Keep "to" before verbs
        if word_lower == 'to':
            if index + 1 < len(words):
                next_word = words[index + 1].lower()
                # Common verbs that need "to"
                if next_word in ['build', 'create', 'implement', 'develop', 'design']:
                    return True
        
        return False

    def _compress_common_phrases(self, text: str) -> str:
        """Compress common phrases aggressively."""
        compressions = {
            'in order to': '2',
            'as well as': '&',
            'such as': 'eg',
            'for example': 'eg',
            'that is': 'ie',
            'and so on': 'etc',
            'and so forth': 'etc',
            'with respect to': 're',
            'with regard to': 're',
            'in terms of': 're',
            'on the other hand': 'otoh',
            'at the same time': 'meanwhile',
            'in addition to': '+',
            'as a result': 'thus',
            'due to the fact that': 'because',
            'in spite of': 'despite',
            'in the event that': 'if',
            'in the case of': 'for',
            'make use of': 'use',
            'take into account': 'consider',
            'come to the conclusion': 'conclude',
            'give consideration to': 'consider'
        }
        
        compressed = text
        for phrase, replacement in compressions.items():
            compressed = re.sub(
                r'\b' + re.escape(phrase) + r'\b',
                replacement,
                compressed,
                flags=re.IGNORECASE
            )
        
        return compressed

    def _apply_vowel_compression(self, text: str) -> str:
        """Apply selective vowel compression to technical terms."""
        # Only compress words longer than 6 characters
        words = text.split()
        compressed_words = []
        
        for word in words:
            if len(word) > 6 and word.isalpha():
                # Check if it's a technical term (has consonant clusters)
                if self._is_technical_term(word):
                    compressed_word = self._compress_vowels(word)
                    compressed_words.append(compressed_word)
                else:
                    compressed_words.append(word)
            else:
                compressed_words.append(word)
        
        return ' '.join(compressed_words)

    def _is_technical_term(self, word: str) -> bool:
        """Check if word appears to be a technical term."""
        # Heuristics for technical terms
        consonant_clusters = ['th', 'sh', 'ch', 'ck', 'ng', 'st', 'nd', 'nt']
        
        word_lower = word.lower()
        cluster_count = sum(1 for cluster in consonant_clusters if cluster in word_lower)
        
        # Technical terms often have multiple consonant clusters
        return cluster_count >= 2

    def _compress_vowels(self, word: str) -> str:
        """Compress vowels in a word while keeping it readable."""
        if len(word) <= 4:
            return word
        
        # Keep first and last characters
        if len(word) <= 6:
            return word
        
        # Remove middle vowels but keep consonants
        compressed = word[0]  # Keep first char
        
        for i in range(1, len(word) - 1):
            char = word[i]
            if char.lower() not in 'aeiou':
                compressed += char
            elif i == 1 or i == len(word) - 2:
                # Keep vowels near start/end
                compressed += char
        
        compressed += word[-1]  # Keep last char
        
        # Don't compress if result is too short or unclear
        if len(compressed) < max(4, len(word) * 0.6):
            return word
        
        return compressed

    def _compress_repetitions(self, text: str) -> str:
        """Compress repeated words and phrases."""
        # Remove duplicate consecutive words
        words = text.split()
        deduped_words = []
        
        for i, word in enumerate(words):
            if i == 0 or word.lower() != words[i-1].lower():
                deduped_words.append(word)
        
        return ' '.join(deduped_words)

    def _apply_syntactic_reduction(self, text: str, context: CompressionContext) -> str:
        """Apply syntactic reduction strategies."""
        compressed = text
        
        # Convert passive to active voice where possible
        compressed = self._reduce_passive_voice(compressed)
        
        # Simplify complex sentences
        compressed = self._simplify_sentences(compressed)
        
        # Remove redundant modifiers
        compressed = self._remove_redundant_modifiers(compressed)
        
        return compressed

    def _reduce_passive_voice(self, text: str) -> str:
        """Convert some passive voice to active."""
        # Simple patterns - in production use proper NLP
        replacements = {
            r'\bis being\s+(\w+ed)\b': r'undergoes \1',
            r'\bwill be\s+(\w+ed)\b': r'will \1',
            r'\bhas been\s+(\w+ed)\b': r'was \1'
        }
        
        compressed = text
        for pattern, replacement in replacements.items():
            compressed = re.sub(pattern, replacement, compressed, flags=re.IGNORECASE)
        
        return compressed

    def _simplify_sentences(self, text: str) -> str:
        """Simplify complex sentence structures."""
        # Remove filler phrases
        fillers = [
            'it is important to note that',
            'it should be noted that',
            'it is worth mentioning that',
            'please note that',
            'as mentioned above',
            'as stated earlier',
            'in other words',
            'to put it simply'
        ]
        
        compressed = text
        for filler in fillers:
            compressed = re.sub(
                r'\b' + re.escape(filler) + r'\s*',
                '',
                compressed,
                flags=re.IGNORECASE
            )
        
        return compressed

    def _remove_redundant_modifiers(self, text: str) -> str:
        """Remove redundant adjectives and adverbs."""
        # Common redundant modifiers
        redundant_patterns = [
            r'\bvery\s+',
            r'\bextremely\s+',
            r'\bquite\s+',
            r'\brather\s+',
            r'\bfairly\s+',
            r'\bpretty\s+',
            r'\bbasically\s+',
            r'\bessentially\s+',
            r'\bactually\s+',
            r'\bobviously\s+'
        ]
        
        compressed = text
        for pattern in redundant_patterns:
            compressed = re.sub(pattern, '', compressed, flags=re.IGNORECASE)
        
        return compressed

    def _apply_global_optimizations(self, text: str, context: CompressionContext) -> str:
        """Apply final global optimizations."""
        compressed = text
        
        # Clean up extra spaces
        compressed = re.sub(r'\s+', ' ', compressed)
        
        # Remove trailing/leading spaces
        compressed = compressed.strip()
        
        # Remove empty lines
        lines = compressed.split('\n')
        non_empty_lines = [line.strip() for line in lines if line.strip()]
        compressed = '\n'.join(non_empty_lines)
        
        return compressed

    def _estimate_quality(self, original: str, compressed: str) -> float:
        """Estimate compression quality."""
        # Simple quality estimation
        original_words = set(original.lower().split())
        compressed_words = set(compressed.lower().split())
        
        if not original_words:
            return 1.0
        
        # Calculate word preservation ratio
        preserved_words = original_words.intersection(compressed_words)
        preservation_ratio = len(preserved_words) / len(original_words)
        
        # Adjust for compression ratio
        compression_ratio = len(compressed) / len(original)
        
        # Balance preservation and compression
        quality = preservation_ratio * 0.7 + (1 - compression_ratio) * 0.3
        
        return min(1.0, quality)
