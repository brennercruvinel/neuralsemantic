"""Engine factory for creating and managing compression engines."""

import logging
from typing import Dict, Type, Optional

from .base_engine import BaseCompressionEngine
from .semantic_engine import SemanticCompressionEngine
from .extreme_engine import ExtremeCompressionEngine
from .hybrid_engine import HybridCompressionEngine
from ..core.types import CompressionLevel, CompressionContext
from ..core.exceptions import ConfigurationError

logger = logging.getLogger(__name__)


class EngineFactory:
    """Factory for creating and managing compression engines."""

    def __init__(self, pattern_manager, vector_store=None, config=None):
        self.pattern_manager = pattern_manager
        self.vector_store = vector_store
        self.config = config
        
        # Engine registry
        self._engine_classes: Dict[str, Type[BaseCompressionEngine]] = {
            'semantic': SemanticCompressionEngine,
            'extreme': ExtremeCompressionEngine,
            'hybrid': HybridCompressionEngine
        }
        
        # Engine instances cache
        self._engine_instances: Dict[str, BaseCompressionEngine] = {}
        
        # Engine selection rules
        self._selection_rules = {
            CompressionLevel.LIGHT: 'semantic',
            CompressionLevel.BALANCED: 'hybrid',
            CompressionLevel.AGGRESSIVE: 'extreme',
            CompressionLevel.NONE: None  # No compression
        }

    def create_engine(self, engine_type: str) -> BaseCompressionEngine:
        """Create a compression engine of the specified type."""
        if engine_type not in self._engine_classes:
            available_engines = ', '.join(self._engine_classes.keys())
            raise ConfigurationError(
                f"Unknown engine type: {engine_type}. "
                f"Available engines: {available_engines}"
            )
        
        # Return cached instance if available
        if engine_type in self._engine_instances:
            return self._engine_instances[engine_type]
        
        # Create new instance
        engine_class = self._engine_classes[engine_type]
        
        try:
            engine = engine_class(
                pattern_manager=self.pattern_manager,
                vector_store=self.vector_store,
                config=self.config
            )
            
            # Cache instance
            self._engine_instances[engine_type] = engine
            
            logger.info(f"Created {engine_type} compression engine")
            return engine
            
        except Exception as e:
            logger.error(f"Failed to create {engine_type} engine: {e}")
            raise ConfigurationError(f"Engine creation failed: {e}") from e

    def get_engine_for_context(self, context: CompressionContext) -> Optional[BaseCompressionEngine]:
        """Get the appropriate engine for the given context."""
        
        # Handle no compression case
        if context.level == CompressionLevel.NONE:
            return None
        
        # Use explicit engine selection if provided
        if hasattr(context, 'engine_preference') and context.engine_preference:
            return self.create_engine(context.engine_preference)
        
        # Use level-based selection
        engine_type = self._selection_rules.get(context.level)
        if engine_type:
            return self.create_engine(engine_type)
        
        # Default to hybrid
        logger.warning(f"No engine mapping for level {context.level}, using hybrid")
        return self.create_engine('hybrid')

    def get_engine_by_name(self, name: str) -> BaseCompressionEngine:
        """Get engine by name, creating if necessary."""
        return self.create_engine(name)

    def register_engine(self, name: str, engine_class: Type[BaseCompressionEngine]) -> None:
        """Register a custom engine class."""
        if not issubclass(engine_class, BaseCompressionEngine):
            raise ConfigurationError(
                f"Engine class must inherit from BaseCompressionEngine"
            )
        
        self._engine_classes[name] = engine_class
        logger.info(f"Registered custom engine: {name}")

    def get_available_engines(self) -> list[str]:
        """Get list of available engine types."""
        return list(self._engine_classes.keys())

    def get_engine_info(self, engine_type: str) -> Dict[str, any]:
        """Get information about an engine type."""
        if engine_type not in self._engine_classes:
            raise ConfigurationError(f"Unknown engine type: {engine_type}")
        
        engine_class = self._engine_classes[engine_type]
        
        return {
            'name': engine_type,
            'class_name': engine_class.__name__,
            'description': engine_class.__doc__ or "No description available",
            'is_instantiated': engine_type in self._engine_instances
        }

    def get_all_engines_info(self) -> Dict[str, Dict[str, any]]:
        """Get information about all available engines."""
        return {
            engine_type: self.get_engine_info(engine_type)
            for engine_type in self._engine_classes.keys()
        }

    def set_engine_selection_rule(self, level: CompressionLevel, engine_type: str) -> None:
        """Set engine selection rule for a compression level."""
        if engine_type and engine_type not in self._engine_classes:
            raise ConfigurationError(f"Unknown engine type: {engine_type}")
        
        self._selection_rules[level] = engine_type
        logger.info(f"Set engine selection rule: {level.value} -> {engine_type}")

    def clear_engine_cache(self) -> None:
        """Clear the engine instance cache."""
        self._engine_instances.clear()
        logger.info("Cleared engine instance cache")

    def warmup_engines(self, engine_types: Optional[list[str]] = None) -> None:
        """Pre-create engine instances for faster access."""
        if engine_types is None:
            engine_types = list(self._engine_classes.keys())
        
        for engine_type in engine_types:
            try:
                self.create_engine(engine_type)
                logger.info(f"Warmed up {engine_type} engine")
            except Exception as e:
                logger.warning(f"Failed to warm up {engine_type} engine: {e}")

    def get_engine_statistics(self) -> Dict[str, any]:
        """Get statistics about engine usage."""
        stats = {
            'total_engines': len(self._engine_classes),
            'instantiated_engines': len(self._engine_instances),
            'available_engines': list(self._engine_classes.keys()),
            'instantiated_engine_types': list(self._engine_instances.keys()),
            'selection_rules': {
                level.value: engine_type 
                for level, engine_type in self._selection_rules.items()
            }
        }
        
        return stats

    def validate_configuration(self) -> Dict[str, any]:
        """Validate engine factory configuration."""
        validation_results = {
            'valid': True,
            'errors': [],
            'warnings': []
        }
        
        # Check if all required engines are available
        required_engines = set(self._selection_rules.values())
        required_engines.discard(None)  # Remove None for NONE level
        
        for engine_type in required_engines:
            if engine_type not in self._engine_classes:
                validation_results['valid'] = False
                validation_results['errors'].append(
                    f"Required engine '{engine_type}' not available"
                )
        
        # Test engine creation
        for engine_type in self._engine_classes.keys():
            try:
                engine = self.create_engine(engine_type)
                if not isinstance(engine, BaseCompressionEngine):
                    validation_results['valid'] = False
                    validation_results['errors'].append(
                        f"Engine '{engine_type}' does not inherit from BaseCompressionEngine"
                    )
            except Exception as e:
                validation_results['warnings'].append(
                    f"Failed to create engine '{engine_type}': {e}"
                )
        
        # Check for missing selection rules
        for level in CompressionLevel:
            if level not in self._selection_rules:
                validation_results['warnings'].append(
                    f"No engine selection rule for compression level: {level.value}"
                )
        
        return validation_results

    def get_recommended_engine(self, text_characteristics: Dict[str, any]) -> str:
        """Get recommended engine based on text characteristics."""
        
        # Extract characteristics
        length = text_characteristics.get('length', 0)
        technical_density = text_characteristics.get('technical_density', 0.0)
        quality_sensitivity = text_characteristics.get('quality_sensitivity', 0.5)
        compression_potential = text_characteristics.get('compression_potential', 0.5)
        
        # Decision logic
        if quality_sensitivity > 0.7:
            return 'semantic'
        elif compression_potential > 0.8 and quality_sensitivity < 0.3:
            return 'extreme'
        elif length > 5000:  # Long text benefits from hybrid approach
            return 'hybrid'
        elif technical_density > 0.6:  # Technical text needs careful handling
            return 'semantic'
        else:
            return 'hybrid'  # Default recommendation

    def benchmark_engines(self, test_texts: list[str], 
                         context: CompressionContext) -> Dict[str, Dict[str, any]]:
        """Benchmark all engines against test texts."""
        results = {}
        
        for engine_type in self._engine_classes.keys():
            engine = self.create_engine(engine_type)
            engine_results = {
                'total_compressions': 0,
                'average_compression_ratio': 0.0,
                'average_processing_time': 0.0,
                'total_errors': 0,
                'compression_ratios': [],
                'processing_times': []
            }
            
            for text in test_texts:
                try:
                    result = engine.compress(text, context)
                    engine_results['compression_ratios'].append(result.compression_ratio)
                    engine_results['processing_times'].append(result.processing_time_ms)
                    engine_results['total_compressions'] += 1
                except Exception as e:
                    engine_results['total_errors'] += 1
                    logger.debug(f"Engine {engine_type} failed on text: {e}")
            
            # Calculate averages
            if engine_results['compression_ratios']:
                engine_results['average_compression_ratio'] = (
                    sum(engine_results['compression_ratios']) / 
                    len(engine_results['compression_ratios'])
                )
            
            if engine_results['processing_times']:
                engine_results['average_processing_time'] = (
                    sum(engine_results['processing_times']) / 
                    len(engine_results['processing_times'])
                )
            
            results[engine_type] = engine_results
        
        return results
