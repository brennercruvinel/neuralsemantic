"""Text processing utilities with multi-model tokenization support."""

import re
import logging
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass

# Try to import tokenizers - graceful degradation if not available
try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

try:
    from transformers import AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    AutoTokenizer = None

logger = logging.getLogger(__name__)


@dataclass
class TokenizationResult:
    """Result of tokenization operation."""
    tokens: List[str]
    token_ids: List[int]
    token_count: int
    model_name: str
    estimated: bool = False


class TokenizerManager:
    """
    Manages multiple tokenizers for accurate token counting across different LLM models.
    Supports GPT-3/4, Claude, Llama, and other models.
    """

    def __init__(self):
        self.tokenizers = {}
        self._initialize_tokenizers()

    def _initialize_tokenizers(self):
        """Initialize available tokenizers."""
        
        # OpenAI tokenizers (GPT-3, GPT-4, etc.)
        if TIKTOKEN_AVAILABLE:
            try:
                self.tokenizers['gpt-4'] = tiktoken.encoding_for_model('gpt-4')
                self.tokenizers['gpt-3.5-turbo'] = tiktoken.encoding_for_model('gpt-3.5-turbo')
                self.tokenizers['text-davinci-003'] = tiktoken.encoding_for_model('text-davinci-003')
                logger.info("OpenAI tokenizers loaded successfully")
            except Exception as e:
                logger.warning(f"Failed to load OpenAI tokenizers: {e}")

        # Claude tokenizer (approximation using GPT-4 tokenizer)
        if TIKTOKEN_AVAILABLE and 'gpt-4' in self.tokenizers:
            self.tokenizers['claude'] = self.tokenizers['gpt-4']  # Approximation
            self.tokenizers['claude-instant'] = self.tokenizers['gpt-4']
            
        # Llama tokenizers
        if TRANSFORMERS_AVAILABLE:
            try:
                # Note: These require downloading models, so we'll add them lazily
                self._llama_tokenizers = {
                    'llama-7b': 'meta-llama/Llama-2-7b-hf',
                    'llama-13b': 'meta-llama/Llama-2-13b-hf',
                    'llama-70b': 'meta-llama/Llama-2-70b-hf'
                }
                logger.info("Transformers available for Llama tokenizers")
            except Exception as e:
                logger.warning(f"Transformers tokenizers setup failed: {e}")

    def get_tokenizer(self, model_name: str):
        """Get tokenizer for specific model."""
        model_name = model_name.lower()
        
        # Direct match
        if model_name in self.tokenizers:
            return self.tokenizers[model_name]
        
        # Fuzzy matching for common model names
        if 'gpt-4' in model_name or 'gpt4' in model_name:
            return self.tokenizers.get('gpt-4')
        elif 'gpt-3.5' in model_name or 'gpt3.5' in model_name:
            return self.tokenizers.get('gpt-3.5-turbo')
        elif 'claude' in model_name:
            return self.tokenizers.get('claude')
        elif 'llama' in model_name:
            return self._get_llama_tokenizer(model_name)
        
        return None

    def _get_llama_tokenizer(self, model_name: str):
        """Get Llama tokenizer with lazy loading."""
        if not TRANSFORMERS_AVAILABLE:
            return None
            
        try:
            # Try to find the best match
            for key, model_id in self._llama_tokenizers.items():
                if key in model_name:
                    if key not in self.tokenizers:
                        logger.info(f"Loading {key} tokenizer...")
                        tokenizer = AutoTokenizer.from_pretrained(model_id)
                        self.tokenizers[key] = tokenizer
                    return self.tokenizers[key]
        except Exception as e:
            logger.warning(f"Failed to load Llama tokenizer: {e}")
            
        return None

    def count_tokens(self, text: str, model: str = 'gpt-4') -> TokenizationResult:
        """
        Count tokens for specific model.
        
        Args:
            text: Text to tokenize
            model: Model name (gpt-4, claude, llama-7b, etc.)
            
        Returns:
            TokenizationResult with token count and metadata
        """
        tokenizer = self.get_tokenizer(model)
        
        if tokenizer is None:
            # Fallback to estimation
            estimated_count = self._estimate_tokens(text)
            return TokenizationResult(
                tokens=[],
                token_ids=[],
                token_count=estimated_count,
                model_name=model,
                estimated=True
            )
        
        try:
            if hasattr(tokenizer, 'encode'):
                # tiktoken or similar interface
                if TIKTOKEN_AVAILABLE and isinstance(tokenizer, tiktoken.Encoding):
                    token_ids = tokenizer.encode(text)
                    tokens = [tokenizer.decode([tid]) for tid in token_ids]
                elif hasattr(tokenizer, 'tokenize'):
                    # Transformers tokenizer
                    tokens = tokenizer.tokenize(text)
                    token_ids = tokenizer.convert_tokens_to_ids(tokens)
                else:
                    # Generic encode interface
                    token_ids = tokenizer.encode(text)
                    tokens = []  # Can't decode individual tokens
                    
                return TokenizationResult(
                    tokens=tokens,
                    token_ids=token_ids,
                    token_count=len(token_ids),
                    model_name=model,
                    estimated=False
                )
            else:
                # Fallback
                estimated_count = self._estimate_tokens(text)
                return TokenizationResult(
                    tokens=[],
                    token_ids=[],
                    token_count=estimated_count,
                    model_name=model,
                    estimated=True
                )
                
        except Exception as e:
            logger.warning(f"Tokenization failed for {model}: {e}")
            estimated_count = self._estimate_tokens(text)
            return TokenizationResult(
                tokens=[],
                token_ids=[],
                token_count=estimated_count,
                model_name=model,
                estimated=True
            )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count using heuristics."""
        # GPT-style estimation: roughly 4 characters per token
        # Adjust for different languages and text types
        
        char_count = len(text)
        word_count = len(text.split())
        
        # Base estimation
        base_estimate = char_count / 4
        
        # Adjustments
        if word_count > 0:
            avg_word_length = char_count / word_count
            if avg_word_length > 8:
                # Longer words typically use more tokens
                base_estimate *= 1.2
            elif avg_word_length < 4:
                # Shorter words might use fewer tokens
                base_estimate *= 0.9
        
        # Technical content adjustment
        if self._is_technical_content(text):
            base_estimate *= 1.1  # Technical terms often use more tokens
        
        return max(1, int(base_estimate))

    def _is_technical_content(self, text: str) -> bool:
        """Check if text appears to be technical content."""
        technical_indicators = [
            'function', 'class', 'import', 'def', 'var', 'const',
            'api', 'http', 'json', 'sql', 'database', 'server',
            'authentication', 'authorization', 'configuration'
        ]
        
        text_lower = text.lower()
        tech_count = sum(1 for indicator in technical_indicators if indicator in text_lower)
        
        return tech_count >= 2

    def get_optimal_compressions_for_model(self, text: str, model: str) -> Dict[str, Any]:
        """
        Get model-specific compression strategies.
        
        Returns suggestions for optimal compression based on tokenizer analysis.
        """
        tokenization = self.count_tokens(text, model)
        
        if tokenization.estimated:
            return {
                'model': model,
                'estimated': True,
                'strategies': ['use general compression patterns']
            }
        
        # Analyze token patterns
        strategies = []
        
        # Find multi-token words that could be compressed
        if tokenization.tokens:
            multi_token_words = self._find_multi_token_words(tokenization.tokens)
            if multi_token_words:
                strategies.append(f"Compress multi-token words: {', '.join(multi_token_words[:5])}")
        
        # Check for compressible patterns
        compressible_patterns = self._find_compressible_patterns(text, tokenization)
        strategies.extend(compressible_patterns)
        
        return {
            'model': model,
            'original_tokens': tokenization.token_count,
            'estimated': tokenization.estimated,
            'strategies': strategies,
            'potential_savings': self._estimate_potential_savings(text, tokenization)
        }

    def _find_multi_token_words(self, tokens: List[str]) -> List[str]:
        """Find words that are split into multiple tokens."""
        # This is a simplified version - in practice, you'd need more sophisticated analysis
        multi_token_candidates = []
        
        i = 0
        while i < len(tokens) - 1:
            # Look for tokens that might be parts of a single word
            current = tokens[i].strip()
            next_token = tokens[i + 1].strip()
            
            if (len(current) > 2 and len(next_token) > 2 and 
                not current.endswith(' ') and not next_token.startswith(' ')):
                combined = current + next_token
                if self._looks_like_single_word(combined):
                    multi_token_candidates.append(combined)
                    i += 2
                    continue
            i += 1
        
        return multi_token_candidates[:10]  # Limit results

    def _looks_like_single_word(self, text: str) -> bool:
        """Check if text looks like it should be a single word."""
        # Simple heuristic
        return (text.isalpha() and 
                len(text) > 5 and 
                text.lower() in ['authentication', 'authorization', 'implementation', 
                                'configuration', 'development', 'application'])

    def _find_compressible_patterns(self, text: str, tokenization: TokenizationResult) -> List[str]:
        """Find patterns that could be compressed efficiently."""
        patterns = []
        
        # Look for repeated phrases
        words = text.split()
        if len(words) > 10:
            # Find 2-3 word phrases that repeat
            phrase_counts = {}
            for i in range(len(words) - 2):
                phrase = ' '.join(words[i:i+3])
                phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
            
            repeated_phrases = [phrase for phrase, count in phrase_counts.items() if count > 1]
            if repeated_phrases:
                patterns.append(f"Compress repeated phrases: {repeated_phrases[0]}")
        
        # Look for verbose constructions
        verbose_patterns = [
            'in order to', 'due to the fact that', 'for the purpose of',
            'with respect to', 'in terms of'
        ]
        
        found_verbose = [pattern for pattern in verbose_patterns if pattern in text.lower()]
        if found_verbose:
            patterns.append(f"Simplify verbose patterns: {', '.join(found_verbose)}")
        
        return patterns

    def _estimate_potential_savings(self, text: str, tokenization: TokenizationResult) -> Dict[str, float]:
        """Estimate potential token savings from compression."""
        if tokenization.estimated:
            return {'estimated_reduction': '20-40%'}
        
        original_tokens = tokenization.token_count
        
        # Estimate savings from different strategies
        savings = {}
        
        # Pattern compression savings
        pattern_savings = len(re.findall(r'\b(authentication|authorization|implementation|configuration)\b', text.lower()))
        savings['pattern_compression'] = (pattern_savings * 2) / original_tokens if original_tokens > 0 else 0
        
        # Verbose phrase savings
        verbose_savings = len(re.findall(r'\b(in order to|due to the fact that|for the purpose of)\b', text.lower()))
        savings['verbose_reduction'] = (verbose_savings * 3) / original_tokens if original_tokens > 0 else 0
        
        # Stop word removal savings
        stop_words = len(re.findall(r'\b(the|a|an|and|or|but|in|on|at|to|for|of|with|by)\b', text.lower()))
        savings['stop_word_removal'] = (stop_words * 0.5) / original_tokens if original_tokens > 0 else 0
        
        total_savings = sum(savings.values())
        savings['total_estimated'] = min(0.7, total_savings)  # Cap at 70%
        
        return savings


class TextProcessor:
    """Advanced text processing utilities."""

    def __init__(self):
        self.tokenizer_manager = TokenizerManager()

    def analyze_text_characteristics(self, text: str) -> Dict[str, Any]:
        """Analyze text characteristics for compression optimization."""
        
        words = text.split()
        sentences = re.split(r'[.!?]+', text)
        
        analysis = {
            'length': len(text),
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(word) for word in words) / len(words) if words else 0,
            'avg_sentence_length': len(words) / len(sentences) if sentences else 0,
            'technical_density': self._calculate_technical_density(text),
            'formality_score': self._calculate_formality_score(text),
            'redundancy_score': self._calculate_redundancy_score(text),
            'compression_potential': self._estimate_compression_potential(text)
        }
        
        return analysis

    def _calculate_technical_density(self, text: str) -> float:
        """Calculate technical content density."""
        technical_terms = [
            'api', 'database', 'server', 'client', 'function', 'method', 'class',
            'algorithm', 'implementation', 'architecture', 'framework', 'library',
            'interface', 'protocol', 'authentication', 'authorization', 'configuration',
            'deployment', 'infrastructure', 'microservice', 'kubernetes', 'docker'
        ]
        
        words = text.lower().split()
        tech_count = sum(1 for word in words if any(term in word for term in technical_terms))
        
        return tech_count / len(words) if words else 0.0

    def _calculate_formality_score(self, text: str) -> float:
        """Calculate text formality score."""
        formal_indicators = [
            'therefore', 'however', 'furthermore', 'consequently', 'nevertheless',
            'moreover', 'subsequently', 'accordingly', 'thus', 'hence'
        ]
        
        informal_indicators = [
            "it's", "don't", "won't", "can't", "isn't", "aren't",
            'yeah', 'okay', 'cool', 'awesome', 'gonna', 'wanna'
        ]
        
        text_lower = text.lower()
        formal_count = sum(1 for indicator in formal_indicators if indicator in text_lower)
        informal_count = sum(1 for indicator in informal_indicators if indicator in text_lower)
        
        if formal_count + informal_count == 0:
            return 0.5  # Neutral
        
        return formal_count / (formal_count + informal_count)

    def _calculate_redundancy_score(self, text: str) -> float:
        """Calculate text redundancy score."""
        words = text.lower().split()
        
        if len(words) < 10:
            return 0.0
        
        # Count repeated words
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Count repeated phrases
        phrase_counts = {}
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
        
        repeated_words = sum(count - 1 for count in word_counts.values() if count > 1)
        repeated_phrases = sum(count - 1 for count in phrase_counts.values() if count > 1)
        
        redundancy = (repeated_words + repeated_phrases * 2) / len(words)
        return min(1.0, redundancy)

    def _estimate_compression_potential(self, text: str) -> float:
        """Estimate how much text can be compressed."""
        factors = []
        
        # Long words can often be abbreviated
        words = text.split()
        long_words = sum(1 for word in words if len(word) > 8)
        factors.append((long_words / len(words)) if words else 0)
        
        # Verbose phrases can be simplified
        verbose_phrases = [
            'in order to', 'due to the fact that', 'for the purpose of',
            'with respect to', 'in terms of', 'as a result of'
        ]
        verbose_count = sum(1 for phrase in verbose_phrases if phrase in text.lower())
        factors.append(min(0.3, verbose_count / len(words)) if words else 0)
        
        # Technical content often has abbreviations available
        tech_density = self._calculate_technical_density(text)
        factors.append(tech_density * 0.4)
        
        # Redundancy can be reduced
        redundancy = self._calculate_redundancy_score(text)
        factors.append(redundancy * 0.3)
        
        return min(1.0, sum(factors))

    def split_preserving_structure(self, text: str) -> List[Dict[str, Any]]:
        """Split text while preserving important structures."""
        segments = []
        
        # Split by paragraphs first
        paragraphs = text.split('\n\n')
        
        for para in paragraphs:
            if not para.strip():
                continue
                
            # Check if paragraph is code
            if self._is_code_block(para):
                segments.append({
                    'text': para,
                    'type': 'code',
                    'compressible': False
                })
            elif self._is_list(para):
                segments.append({
                    'text': para,
                    'type': 'list',
                    'compressible': True
                })
            elif self._is_heading(para):
                segments.append({
                    'text': para,
                    'type': 'heading',
                    'compressible': True
                })
            else:
                segments.append({
                    'text': para,
                    'type': 'paragraph',
                    'compressible': True
                })
        
        return segments

    def _is_code_block(self, text: str) -> bool:
        """Check if text appears to be a code block."""
        code_indicators = [
            'function', 'def ', 'class ', 'import ', 'from ',
            'return ', '=>', '->', '{', '}', '()', '[]'
        ]
        
        # Check for indentation patterns
        lines = text.split('\n')
        indented_lines = sum(1 for line in lines if line.startswith('    ') or line.startswith('\t'))
        
        # Check for code keywords
        code_keywords = sum(1 for indicator in code_indicators if indicator in text.lower())
        
        return indented_lines > len(lines) * 0.5 or code_keywords >= 2

    def _is_list(self, text: str) -> bool:
        """Check if text appears to be a list."""
        lines = text.split('\n')
        list_lines = sum(1 for line in lines if re.match(r'^\s*[-*•]\s+', line) or 
                        re.match(r'^\s*\d+\.\s+', line))
        
        return list_lines > len(lines) * 0.5

    def _is_heading(self, text: str) -> bool:
        """Check if text appears to be a heading."""
        text = text.strip()
        
        # Markdown headings
        if text.startswith('#'):
            return True
        
        # Short text that ends with colon
        if len(text) < 100 and text.endswith(':'):
            return True
        
        # All caps (but not too long)
        if text.isupper() and len(text) < 50:
            return True
        
        return False

    def clean_text(self, text: str) -> str:
        """Clean text for better compression."""
        cleaned = text
        
        # Normalize whitespace
        cleaned = re.sub(r'\s+', ' ', cleaned)
        
        # Remove extra punctuation
        cleaned = re.sub(r'[.]{2,}', '...', cleaned)
        cleaned = re.sub(r'[!]{2,}', '!', cleaned)
        cleaned = re.sub(r'[?]{2,}', '?', cleaned)
        
        # Clean up quotes
        cleaned = re.sub(r'["""]', '"', cleaned)
        cleaned = re.sub(r"[''']", "'", cleaned)
        
        return cleaned.strip()
