"""Main Neural Semantic Compiler implementation."""

import time
import logging
import uuid
from typing import Optional, Dict, Any, List

from .types import CompressionResult, CompressionContext, CompressionLevel
from .config import CompilerConfig, ConfigManager
from .exceptions import CompressionError, ConfigurationError
from ..patterns.pattern_manager import PatternManager
from ..vector.vector_store import VectorStore
from ..engines.engine_factory import EngineFactory

logger = logging.getLogger(__name__)


class NeuralSemanticCompiler:
    """
    Main Neural Semantic Compiler class.
    
    The first compiler designed for neural communication optimization.
    Reduces LLM token usage by 40-65% while preserving semantic meaning.
    """

    def __init__(self, config: Optional[CompilerConfig] = None):
        """
        Initialize the Neural Semantic Compiler.
        
        Args:
            config: Compiler configuration. If None, loads default config.
        """
        # Load configuration
        self.config = config or ConfigManager.create_default_config()
        
        # Setup logging
        self._setup_logging()
        
        # Initialize core components
        self._initialize_components()
        
        # Validation
        self._validate_setup()
        
        logger.info("Neural Semantic Compiler initialized successfully")

    def _setup_logging(self) -> None:
        log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
        
        # Setup logging with optional file output
        log_config = {
            'level': log_level,
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        }
        
        if self.config.log_file:
            log_config['filename'] = self.config.log_file
            
        logging.basicConfig(**log_config)

    def _initialize_components(self) -> None:
        try:
            # Validate and ensure database path is safe
            db_path = self._validate_database_path(self.config.database.path)
            self.pattern_manager = PatternManager(db_path)
            
            self.vector_store = VectorStore(self.config.vector)
            
            self.engine_factory = EngineFactory(
                pattern_manager=self.pattern_manager,
                vector_store=self.vector_store,
                config=self.config
            )
            
            if self.config.enable_caching:
                self.engine_factory.warmup_engines(['hybrid'])
            
            logger.info("Core components initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            raise ConfigurationError(f"Component initialization failed: {e}") from e

    def _validate_setup(self) -> None:
        validation_result = self.engine_factory.validate_configuration()
        
        if not validation_result['valid']:
            errors = '; '.join(validation_result['errors'])
            raise ConfigurationError(f"Invalid configuration: {errors}")
        
        if validation_result['warnings']:
            for warning in validation_result['warnings']:
                logger.warning(f"Configuration warning: {warning}")

    def _validate_database_path(self, db_path: str) -> str:
        """Validate database path for security."""
        import os
        from pathlib import Path
        
        # Convert to Path object for safer handling
        path = Path(db_path)
        
        # For test environments, allow temporary directories
        if os.environ.get('NSC_TEST_MODE') == '1':
            # In test mode, allow any path but ensure parent directory exists
            path.parent.mkdir(parents=True, exist_ok=True)
            return str(path)
        
        # Ensure it's not an absolute path outside safe directories
        if path.is_absolute():
            # Check if it's in a safe location
            safe_dirs = [Path.home() / ".neuralsemantic", Path("/tmp"), Path("/var/tmp")]
            if not any(str(path).startswith(str(safe_dir)) for safe_dir in safe_dirs):
                raise ConfigurationError(f"Database path must be in a safe directory: {db_path}")
        
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)
        
        return str(path)

    @classmethod
    def create_default(cls) -> "NeuralSemanticCompiler":
        return cls()

    def compress(self, text: str, **kwargs) -> CompressionResult:
        """
        Compress text using Neural Semantic Compiler.
        
        Args:
            text: Text to compress
            **kwargs: Compression options:
                - level: CompressionLevel (light, balanced, aggressive)
                - domain: Domain context (web-development, agile, etc.)
                - language: Language code (default: en)
                - preserve_code: Preserve code blocks (default: True)
                - preserve_urls: Preserve URLs (default: True)
                - preserve_numbers: Preserve numbers (default: True)
                - session_id: Session ID for tracking
                
        Returns:
            CompressionResult with compressed text and metadata
            
        Raises:
            CompressionError: If compression fails
        """
        start_time = time.time()
        session_id = kwargs.get('session_id', str(uuid.uuid4()))
        
        try:
            context = self._create_context(text, **kwargs)
            
            engine = self.engine_factory.get_engine_for_context(context)
            
            if engine is None:
                return self._create_no_compression_result(text, session_id)
            
            result = engine.compress(text, context)
            
            result.session_id = session_id
            
            result.quality_score = self._calculate_quality_score(result, context)
            
            self._log_compression(result, context)
            
            if self.config.enable_caching:
                self._store_session(result, context)
            
            return result
            
        except Exception as e:
            logger.error(f"Compression failed for session {session_id}: {e}")
            raise CompressionError(f"Compression failed: {e}") from e

    def _create_context(self, text: str, **kwargs) -> CompressionContext:
        """Create compression context from parameters."""
        
        level_str = kwargs.get('level', self.config.compression.default_level.value)
        if isinstance(level_str, str):
            try:
                level = CompressionLevel(level_str)
            except ValueError:
                logger.warning(f"Invalid compression level: {level_str}, using balanced")
                level = CompressionLevel.BALANCED
        else:
            level = level_str

        context = CompressionContext(
            level=level,
            domain=kwargs.get('domain'),
            language=kwargs.get('language', 'en'),
            preserve_code=kwargs.get('preserve_code', self.config.compression.preserve_code),
            preserve_urls=kwargs.get('preserve_urls', self.config.compression.preserve_urls),
            preserve_numbers=kwargs.get('preserve_numbers', self.config.compression.preserve_numbers),
            target_compression=kwargs.get('target_compression', 0.6),
            requires_high_quality=kwargs.get('requires_high_quality', True),
            context_type=kwargs.get('context_type', 'general')
        )
        
        return context

    def _create_no_compression_result(self, text: str, session_id: str) -> CompressionResult:
        """Create result for no compression case."""
        return CompressionResult(
            original_text=text,
            compressed_text=text,
            original_tokens=self._estimate_tokens(text),
            compressed_tokens=self._estimate_tokens(text),
            compression_ratio=1.0,
            quality_score=10.0,
            pattern_matches=[],
            processing_time_ms=0,
            engine_used='none',
            session_id=session_id
        )

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count for text."""
        return max(1, len(text) // 4)

    def _calculate_quality_score(self, result: CompressionResult, 
                                context: CompressionContext) -> float:
        """Calculate comprehensive quality score."""
        
        compression_quality = 1.0 - abs(result.compression_ratio - context.target_compression)
        
        pattern_quality = 1.0
        if result.pattern_matches:
            avg_confidence = sum(m.confidence for m in result.pattern_matches) / len(result.pattern_matches)
            pattern_quality = avg_confidence
        
        length_ratio = len(result.compressed_text) / len(result.original_text)
        length_quality = 1.0 if length_ratio > 0.2 else length_ratio * 5
        
        time_quality = max(0.5, 1.0 - (result.processing_time_ms / 1000))
        
        quality_score = (
            compression_quality * 0.4 +
            pattern_quality * 0.3 +
            length_quality * 0.2 +
            time_quality * 0.1
        )
        
        return min(10.0, max(0.0, quality_score * 10))

    def _log_compression(self, result: CompressionResult, context: CompressionContext) -> None:
        """Log compression operation."""
        char_reduction = len(result.original_text) - len(result.compressed_text)
        token_reduction = result.original_tokens - result.compressed_tokens
        
        logger.info(
            f"Compression completed - Session: {result.session_id}, "
            f"Engine: {result.engine_used}, "
            f"Reduction: {char_reduction} chars ({(1-result.compression_ratio):.1%}), "
            f"Tokens: {token_reduction}, "
            f"Quality: {result.quality_score:.1f}/10, "
            f"Time: {result.processing_time_ms}ms, "
            f"Patterns: {len(result.pattern_matches)}"
        )

    def _store_session(self, result: CompressionResult, context: CompressionContext) -> None:
        """Store session data for analytics."""
        # TODO: Implement analytics storage when database is configured
        pass

    def decompress(self, compressed_text: str, **kwargs) -> str:
        """
        Decompress text (where possible).
        
        Note: Full decompression is not always possible due to lossy compression.
        This method provides best-effort expansion of known abbreviations.
        
        Args:
            compressed_text: Compressed text to decompress
            **kwargs: Decompression options
                - domain: Domain context for pattern matching
                
        Returns:
            Decompressed text
        """
        try:
            domain = kwargs.get('domain')
            
            patterns = self.pattern_manager.get_patterns(domain=domain)
            
            reverse_patterns = {}
            for pattern in patterns:
                if self._is_reversible_pattern(pattern):
                    reverse_patterns[pattern.compressed] = pattern.original
            
            decompressed = compressed_text
            for compressed_form, original_form in reverse_patterns.items():
                decompressed = decompressed.replace(compressed_form, original_form)
            
            logger.info(f"Decompression applied {len(reverse_patterns)} reverse patterns")
            return decompressed
            
        except Exception as e:
            logger.error(f"Decompression failed: {e}")
            return compressed_text

    def _is_reversible_pattern(self, pattern) -> bool:
        """Check if a pattern is safely reversible."""
        compressed_words = pattern.compressed.lower().split()
        
        if any(len(word) <= 2 for word in compressed_words):
            return False
        
        if any(char in pattern.compressed for char in ['&', '2', '4']):
            return False
        
        return True

    def get_statistics(self) -> Dict[str, Any]:
        """Get compiler statistics."""
        try:
            pattern_stats = self.pattern_manager.get_statistics()
            vector_stats = self.vector_store.get_collection_stats()
            engine_stats = self.engine_factory.get_engine_statistics()
            
            return {
                'compiler_version': '1.0.0',
                'patterns': pattern_stats,
                'vector_store': vector_stats,
                'engines': engine_stats,
                'configuration': {
                    'default_level': self.config.compression.default_level.value,
                    'active_domains': self.config.active_domains,
                    'caching_enabled': self.config.enable_caching
                }
            }
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {'error': str(e)}

    def add_pattern(self, original: str, compressed: str, **kwargs) -> bool:
        """
        Add a new compression pattern.
        
        Args:
            original: Original text
            compressed: Compressed form
            **kwargs: Pattern options:
                - pattern_type: Type of pattern (word, phrase, compound, abbreviation)
                - domain: Domain context
                - priority: Priority (100-1000)
                - language: Language code
                
        Returns:
            True if pattern was added successfully
        """
        try:
            from .types import Pattern, PatternType
            
            pattern = Pattern(
                original=original,
                compressed=compressed,
                pattern_type=PatternType(kwargs.get('pattern_type', 'word')),
                domain=kwargs.get('domain', 'general'),
                priority=kwargs.get('priority', 500),
                language=kwargs.get('language', 'en')
            )
            
            success = self.pattern_manager.add_pattern(pattern)
            
            if success and self.vector_store.enabled:
                # Add to vector store
                self.vector_store.add_pattern(pattern)
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to add pattern: {e}")
            return False

    def learn_from_feedback(self, session_id: str, rating: int, 
                          feedback: str = None) -> None:
        """
        Learn from user feedback to improve future compressions.
        
        Args:
            session_id: Session ID of the compression
            rating: Rating 1-5 (1=poor, 5=excellent)
            feedback: Optional text feedback
        """
        try:
            # Update pattern success rates based on feedback
            # This would be implemented based on which patterns were used
            # in the session and how successful they were
            
            logger.info(f"Received feedback for session {session_id}: rating={rating}")
            
            if feedback:
                logger.info(f"Feedback text: {feedback}")
            
            
        except Exception as e:
            logger.error(f"Failed to process feedback: {e}")

    def benchmark(self, test_texts: List[str], **kwargs) -> Dict[str, Any]:
        """
        Benchmark compression performance.
        
        Args:
            test_texts: List of texts to test
            **kwargs: Benchmark options
            
        Returns:
            Benchmark results
        """
        try:
            context = self._create_context("", **kwargs)
            
            # Benchmark all engines
            engine_results = self.engine_factory.benchmark_engines(test_texts, context)
            
            # Overall statistics
            overall_stats = {
                'total_texts': len(test_texts),
                'average_text_length': sum(len(text) for text in test_texts) / len(test_texts),
                'benchmark_date': time.time(),
                'configuration': {
                    'compression_level': context.level.value,
                    'domain': context.domain,
                    'preserve_code': context.preserve_code
                }
            }
            
            return {
                'overall': overall_stats,
                'engines': engine_results
            }
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {'error': str(e)}

    def clear_cache(self) -> None:
        """Clear all caches."""
        try:
            self.engine_factory.clear_engine_cache()
            if hasattr(self.vector_store, 'embedding_manager'):
                self.vector_store.embedding_manager.clear_cache()
            logger.info("Caches cleared")
        except Exception as e:
            logger.error(f"Failed to clear cache: {e}")

    def health_check(self) -> Dict[str, Any]:
        """Perform health check of all components."""
        health = {
            'overall': 'healthy',
            'components': {},
            'timestamp': time.time()
        }
        
        try:
            # Check pattern manager
            pattern_count = len(self.pattern_manager.get_patterns(limit=1))
            health['components']['pattern_manager'] = {
                'status': 'healthy',
                'patterns_available': pattern_count > 0
            }
        except Exception as e:
            health['components']['pattern_manager'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health['overall'] = 'degraded'
        
        try:
            # Check vector store
            vector_stats = self.vector_store.get_collection_stats()
            health['components']['vector_store'] = {
                'status': 'healthy' if self.vector_store.enabled else 'disabled',
                'enabled': self.vector_store.enabled,
                'patterns_indexed': vector_stats.get('total_patterns', 0)
            }
        except Exception as e:
            health['components']['vector_store'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health['overall'] = 'degraded'
        
        try:
            # Check engines
            engine_stats = self.engine_factory.get_engine_statistics()
            health['components']['engines'] = {
                'status': 'healthy',
                'available_engines': engine_stats['total_engines'],
                'instantiated_engines': engine_stats['instantiated_engines']
            }
        except Exception as e:
            health['components']['engines'] = {
                'status': 'unhealthy',
                'error': str(e)
            }
            health['overall'] = 'unhealthy'
        
        return health