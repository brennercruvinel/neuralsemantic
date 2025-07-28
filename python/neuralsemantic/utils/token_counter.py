"""Token counting utilities for Neural Semantic Compiler."""

import logging
from typing import Dict, Any, Optional

try:
    import tiktoken
    TIKTOKEN_AVAILABLE = True
except ImportError:
    TIKTOKEN_AVAILABLE = False
    tiktoken = None

logger = logging.getLogger(__name__)


class TokenCounter:
    """
    Token counting utility that supports multiple LLM tokenizers.
    
    Provides accurate token counting for different models to ensure
    precise compression ratio calculations.
    """

    def __init__(self, default_model: str = "gpt-4"):
        self.default_model = default_model
        self._encoding_cache: Dict[str, Any] = {}
        self._initialize_encoders()

    def _initialize_encoders(self) -> None:
        """Initialize token encoders."""
        if TIKTOKEN_AVAILABLE:
            try:
                # Cache common encodings
                common_models = ["gpt-4", "gpt-3.5-turbo", "text-davinci-003"]
                for model in common_models:
                    try:
                        encoding = tiktoken.encoding_for_model(model)
                        self._encoding_cache[model] = encoding
                    except Exception:
                        # Model not found, skip
                        continue
                        
                # Also cache by encoding name
                common_encodings = ["cl100k_base", "p50k_base", "r50k_base"]
                for enc_name in common_encodings:
                    try:
                        encoding = tiktoken.get_encoding(enc_name)
                        self._encoding_cache[enc_name] = encoding
                    except Exception:
                        continue
                        
                logger.info(f"Initialized {len(self._encoding_cache)} token encoders")
                
            except Exception as e:
                logger.warning(f"Failed to initialize some encoders: {e}")
        else:
            logger.warning("tiktoken not available, using fallback token counting")

    def count_tokens(self, text: str, model: Optional[str] = None) -> int:
        """
        Count tokens in text for specified model.
        
        Args:
            text: Input text
            model: Model name (e.g., 'gpt-4', 'gpt-3.5-turbo')
            
        Returns:
            Number of tokens
        """
        if not text:
            return 0
            
        model = model or self.default_model
        
        if TIKTOKEN_AVAILABLE and model in self._encoding_cache:
            try:
                encoding = self._encoding_cache[model]
                return len(encoding.encode(text))
            except Exception as e:
                logger.warning(f"Tiktoken encoding failed for {model}: {e}")
        
        # Fallback to approximation
        return self._approximate_token_count(text)

    def _approximate_token_count(self, text: str) -> int:
        """
        Approximate token count when tiktoken is not available.
        
        Uses the rough heuristic: 1 token ≈ 4 characters for English text.
        This is less accurate but provides a reasonable estimate.
        """
        # Basic approximation: 1 token ≈ 4 characters
        char_count = len(text)
        
        # Adjust for language patterns
        word_count = len(text.split())
        
        # Heuristic: Most tokens are partial words
        # Average English word is ~5 chars, ~1.3 tokens
        if word_count > 0:
            estimated_tokens = int(word_count * 1.3)
        else:
            estimated_tokens = char_count // 4
        
        # Minimum of 1 token for non-empty text
        return max(1, estimated_tokens) if text.strip() else 0

    def compare_models(self, text: str, models: list[str]) -> Dict[str, int]:
        """Compare token counts across multiple models."""
        results = {}
        
        for model in models:
            try:
                results[model] = self.count_tokens(text, model)
            except Exception as e:
                logger.warning(f"Failed to count tokens for {model}: {e}")
                results[model] = self._approximate_token_count(text)
        
        return results

    def get_token_efficiency(self, original: str, compressed: str, 
                           model: Optional[str] = None) -> Dict[str, Any]:
        """Calculate token efficiency metrics."""
        original_tokens = self.count_tokens(original, model)
        compressed_tokens = self.count_tokens(compressed, model)
        
        if original_tokens == 0:
            return {
                "original_tokens": 0,
                "compressed_tokens": 0,
                "token_savings": 0,
                "compression_ratio": 1.0,
                "efficiency_ratio": 0.0
            }
        
        token_savings = original_tokens - compressed_tokens
        compression_ratio = compressed_tokens / original_tokens
        
        # Character efficiency
        char_reduction = len(original) - len(compressed)
        char_ratio = char_reduction / len(original) if len(original) > 0 else 0
        
        # Efficiency: token reduction per character reduction
        efficiency_ratio = token_savings / char_reduction if char_reduction > 0 else 0
        
        return {
            "original_tokens": original_tokens,
            "compressed_tokens": compressed_tokens,
            "token_savings": token_savings,
            "compression_ratio": compression_ratio,
            "savings_percentage": (1 - compression_ratio) * 100,
            "character_reduction": char_reduction,
            "character_ratio": char_ratio,
            "efficiency_ratio": efficiency_ratio,
            "model_used": model or self.default_model
        }

    def estimate_cost_savings(self, original: str, compressed: str,
                            cost_per_1k_tokens: float = 0.03,
                            model: Optional[str] = None) -> Dict[str, float]:
        """Estimate cost savings from compression."""
        efficiency = self.get_token_efficiency(original, compressed, model)
        
        original_cost = (efficiency["original_tokens"] / 1000) * cost_per_1k_tokens
        compressed_cost = (efficiency["compressed_tokens"] / 1000) * cost_per_1k_tokens
        savings = original_cost - compressed_cost
        
        return {
            "original_cost": original_cost,
            "compressed_cost": compressed_cost,
            "cost_savings": savings,
            "savings_percentage": (savings / original_cost * 100) if original_cost > 0 else 0,
            "cost_per_1k_tokens": cost_per_1k_tokens
        }

    def analyze_text_characteristics(self, text: str) -> Dict[str, Any]:
        """Analyze text characteristics relevant to tokenization."""
        words = text.split()
        
        # Basic statistics
        char_count = len(text)
        word_count = len(words)
        sentence_count = len([s for s in text.split('.') if s.strip()])
        
        # Character distribution
        alpha_chars = sum(1 for c in text if c.isalpha())
        digit_chars = sum(1 for c in text if c.isdigit())
        space_chars = sum(1 for c in text if c.isspace())
        punct_chars = char_count - alpha_chars - digit_chars - space_chars
        
        # Word length distribution
        word_lengths = [len(word) for word in words] if words else []
        avg_word_length = sum(word_lengths) / len(word_lengths) if word_lengths else 0
        
        # Tokenization complexity indicators
        camel_case_count = sum(1 for word in words if any(c.isupper() for c in word[1:]))
        special_chars = sum(1 for c in text if not (c.isalnum() or c.isspace()))
        
        return {
            "character_count": char_count,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "average_word_length": avg_word_length,
            "character_distribution": {
                "alphabetic": alpha_chars / char_count if char_count > 0 else 0,
                "numeric": digit_chars / char_count if char_count > 0 else 0,
                "whitespace": space_chars / char_count if char_count > 0 else 0,
                "punctuation": punct_chars / char_count if char_count > 0 else 0
            },
            "complexity_indicators": {
                "camel_case_words": camel_case_count,
                "special_characters": special_chars,
                "lexical_diversity": len(set(words)) / len(words) if words else 0
            }
        }

    def get_supported_models(self) -> list[str]:
        """Get list of supported tokenizer models."""
        supported = list(self._encoding_cache.keys())
        
        if TIKTOKEN_AVAILABLE:
            # Add known models even if not cached
            known_models = [
                "gpt-4", "gpt-4-32k", "gpt-3.5-turbo", "gpt-3.5-turbo-16k",
                "text-davinci-003", "text-davinci-002", "code-davinci-002"
            ]
            supported.extend([m for m in known_models if m not in supported])
        
        return sorted(list(set(supported)))

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get tokenizer cache statistics."""
        return {
            "tiktoken_available": TIKTOKEN_AVAILABLE,
            "cached_encoders": len(self._encoding_cache),
            "default_model": self.default_model,
            "supported_models": len(self.get_supported_models())
        }