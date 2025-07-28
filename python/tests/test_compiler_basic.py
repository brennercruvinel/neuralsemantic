"""
Basic test suite for Neural Semantic Compiler without vector store dependencies.
"""

import pytest
import tempfile
import os

from neuralsemantic.core.compiler import NeuralSemanticCompiler
from neuralsemantic.core.config import CompilerConfig, DatabaseConfig
from neuralsemantic.core.types import CompressionLevel


class TestNeuralSemanticCompilerBasic:
    """Basic tests for the Neural Semantic Compiler without vector store."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        yield db_path
        
        # Cleanup
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def test_config(self, temp_db):
        """Create test configuration."""
        return CompilerConfig(
            database=DatabaseConfig(path=temp_db),
            log_level="ERROR"  # Reduce noise in tests
        )

    @pytest.fixture
    def compiler(self, test_config):
        """Create compiler instance for testing."""
        return NeuralSemanticCompiler(test_config)

    def test_initialization(self, compiler):
        """Test compiler initialization without vector store."""
        assert compiler is not None

    def test_health_check_structure(self, compiler):
        """Test that health check returns proper structure."""
        health = compiler.health_check()
        assert isinstance(health, dict)
        assert 'overall' in health
        assert 'components' in health

    def test_compression_levels_enum(self):
        """Test that CompressionLevel enum is properly defined."""
        assert hasattr(CompressionLevel, 'LIGHT')
        assert hasattr(CompressionLevel, 'BALANCED') 
        assert hasattr(CompressionLevel, 'AGGRESSIVE')

    def test_config_creation(self, temp_db):
        """Test configuration creation."""
        config = CompilerConfig(
            database=DatabaseConfig(path=temp_db),
            log_level="DEBUG"
        )
        assert config.database.path == temp_db
        assert config.log_level == "DEBUG"

    def test_compiler_creation_with_config(self, test_config):
        """Test compiler creation with custom config."""
        compiler = NeuralSemanticCompiler(test_config)
        assert compiler is not None

    def test_compression_basic(self, compiler):
        """Test basic compression functionality."""
        text = "This is a test message for compression"
        result = compiler.compress(text)
        
        assert result is not None
        assert result.original_text == text
        assert result.compressed_text is not None
        assert 0 < result.compression_ratio <= 1
        
    def test_compression_empty_text(self, compiler):
        """Test compression with empty text."""
        with pytest.raises(Exception):
            compiler.compress("")
            
    def test_compression_with_code_preservation(self, compiler):
        """Test compression preserves code when configured."""
        code_text = "```python\ndef test():\n    return True\n```"
        result = compiler.compress(code_text, preserve_code=True)
        
        assert "def test():" in result.compressed_text