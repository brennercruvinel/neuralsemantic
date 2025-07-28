"""Configuration management for Neural Semantic Compiler."""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional
from pydantic import BaseModel, Field

from .types import CompressionLevel


class DatabaseConfig(BaseModel):
    """Database configuration."""
    path: str = "data/patterns.db"
    connection_pool_size: int = 5
    enable_wal_mode: bool = True
    cache_size_mb: int = 64


class VectorConfig(BaseModel):
    """Vector store configuration."""
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    persist_directory: str = "data/vector_store"
    similarity_threshold: float = 0.8
    max_results: int = 10
    enable_gpu: bool = False


class CompressionConfig(BaseModel):
    """Compression configuration."""
    default_level: CompressionLevel = CompressionLevel.BALANCED
    preserve_code: bool = True
    preserve_urls: bool = True
    preserve_numbers: bool = True
    min_compression_ratio: float = 0.1
    max_compression_ratio: float = 0.8
    semantic_threshold: float = 0.90
    target_semantic_score: float = 0.95


class LearningConfig(BaseModel):
    """Learning configuration."""
    enable_auto_discovery: bool = True
    min_pattern_frequency: int = 3
    pattern_quality_threshold: float = 7.0
    feedback_learning_rate: float = 0.1


class CompilerConfig(BaseModel):
    """Main compiler configuration."""
    database: DatabaseConfig = DatabaseConfig()
    vector: VectorConfig = VectorConfig()
    compression: CompressionConfig = CompressionConfig()
    learning: LearningConfig = LearningConfig()

    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None

    # Performance settings
    enable_caching: bool = True
    cache_ttl_seconds: int = 300
    max_cache_size: int = 1000

    # Domain-specific settings
    active_domains: List[str] = ["general", "web-development", "agile"]
    domain_weights: Dict[str, float] = {
        "general": 1.0,
        "web-development": 1.2,
        "agile": 1.1
    }


class ConfigManager:
    """Configuration management with environment variable support."""

    @classmethod
    def load_config(cls, config_path: Optional[str] = None) -> CompilerConfig:
        """Load configuration from file or environment."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return CompilerConfig(**config_data)

        # Load from environment variables
        return cls._load_from_env()

    @classmethod
    def _load_from_env(cls) -> CompilerConfig:
        """Load configuration from environment variables."""
        config_data = {}

        # Database settings
        if os.getenv('NSC_DATABASE_PATH'):
            config_data.setdefault('database', {})['path'] = os.getenv('NSC_DATABASE_PATH')

        # Vector settings
        if os.getenv('NSC_VECTOR_MODEL'):
            config_data.setdefault('vector', {})['model_name'] = os.getenv('NSC_VECTOR_MODEL')

        # Compression settings
        if os.getenv('NSC_COMPRESSION_LEVEL'):
            level = os.getenv('NSC_COMPRESSION_LEVEL').upper()
            if hasattr(CompressionLevel, level):
                config_data.setdefault('compression', {})['default_level'] = getattr(CompressionLevel, level)

        return CompilerConfig(**config_data)

    @classmethod
    def get_default_data_dir(cls) -> Path:
        """Get default data directory."""
        home = Path.home()
        data_dir = home / ".neuralsemantic"
        data_dir.mkdir(exist_ok=True)
        return data_dir

    @classmethod
    def create_default_config(cls) -> CompilerConfig:
        """Create default configuration with proper paths."""
        data_dir = cls.get_default_data_dir()
        
        config = CompilerConfig()
        config.database.path = str(data_dir / "patterns.db")
        config.vector.persist_directory = str(data_dir / "vector_store")
        
        return config