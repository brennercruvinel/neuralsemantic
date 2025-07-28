"""
Test suite for Neural Semantic Compiler
"""

import pytest
import tempfile
import os
from pathlib import Path

from neuralsemantic import NeuralSemanticCompiler, CompressionLevel
from neuralsemantic.core.config import CompilerConfig, DatabaseConfig
from neuralsemantic.core.exceptions import CompressionError


class TestNeuralSemanticCompiler:
    """Test the main Neural Semantic Compiler."""

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
        """Test compiler initialization."""
        assert compiler is not None
        assert hasattr(compiler, 'pattern_manager')
        assert hasattr(compiler, 'engine_factory')

    def test_basic_compression(self, compiler):
        """Test basic compression functionality."""
        text = "Build a production-ready React application with authentication"
        result = compiler.compress(text)

        assert result.original_text == text
        assert result.compressed_text != text
        assert result.compression_ratio < 1.0
        assert result.quality_score > 0
        assert result.original_tokens > 0
        assert result.compressed_tokens > 0
        assert result.engine_used in ['SemanticEngine', 'HybridEngine', 'ExtremeEngine']

    def test_compression_levels(self, compiler):
        """Test different compression levels."""
        text = "Implement user authentication and authorization system"

        light_result = compiler.compress(text, level=CompressionLevel.LIGHT)
        balanced_result = compiler.compress(text, level=CompressionLevel.BALANCED)
        aggressive_result = compiler.compress(text, level=CompressionLevel.AGGRESSIVE)

        # Aggressive should compress more than balanced
        assert aggressive_result.compression_ratio <= balanced_result.compression_ratio
        assert balanced_result.compression_ratio <= light_result.compression_ratio

    def test_domain_specific_compression(self, compiler):
        """Test domain-specific compression."""
        web_text = "Create React components with TypeScript interfaces"
        agile_text = "Sprint planning meeting with product owner"

        web_result = compiler.compress(web_text, domain="web-development")
        agile_result = compiler.compress(agile_text, domain="agile")

        assert web_result.compressed_text != web_text
        assert agile_result.compressed_text != agile_text

    def test_empty_text_handling(self, compiler):
        """Test handling of empty text."""
        with pytest.raises(CompressionError):
            compiler.compress("")

        with pytest.raises(CompressionError):
            compiler.compress(None)

    def test_very_short_text(self, compiler):
        """Test compression of very short text."""
        text = "Hello"
        result = compiler.compress(text)

        # Short text might not compress much
        assert result.original_text == text
        assert result.compressed_tokens <= result.original_tokens

    def test_code_preservation(self, compiler):
        """Test that code blocks are preserved."""
        code_text = '''
        function authenticate(user) {
            return user.password === 'secret123';
        }
        '''

        result = compiler.compress(code_text, preserve_code=True)

        # Code should be minimally compressed
        assert 'function' in result.compressed_text
        assert 'authenticate' in result.compressed_text

    def test_pattern_addition(self, compiler):
        """Test adding custom patterns."""
        success = compiler.add_pattern(
            original="machine learning",
            compressed="ML",
            pattern_type="compound",
            domain="ai",
            priority=800
        )

        assert success is True

        # Test that the pattern is used
        text = "Train a machine learning model"
        result = compiler.compress(text, domain="ai")

        assert "ML" in result.compressed_text

    def test_statistics(self, compiler):
        """Test statistics collection."""
        # Perform some compressions
        texts = [
            "Build React application",
            "Implement user authentication",
            "Create database schema"
        ]

        for text in texts:
            compiler.compress(text)

        stats = compiler.get_statistics()

        assert stats['session']['compressions'] == len(texts)
        assert stats['session']['total_input_tokens'] > 0
        assert stats['session']['total_output_tokens'] > 0
        assert 'patterns' in stats
        assert 'engines' in stats

    def test_session_report(self, compiler):
        """Test session report generation."""
        # Perform compression
        compiler.compress("Build a React application with authentication")

        report = compiler.get_session_report()

        assert "Neural Semantic Compiler" in report
        assert "Session Report" in report
        assert "compressions: 1" in report

    def test_system_health(self, compiler):
        """Test system health validation."""
        health = compiler.validate_system_health()

        assert health['overall_status'] in ['healthy', 'warning', 'error']
        assert 'components' in health
        assert 'pattern_manager' in health['components']
        assert 'engines' in health['components']

    def test_compression_explanation(self, compiler):
        """Test compression explanation."""
        text = "Build production-ready React application"
        explanation = compiler.explain_compression(text)

        assert explanation['input']['text'] == text
        assert explanation['output']['text'] != text
        assert 'compression' in explanation
        assert 'engine' in explanation
        assert 'patterns_applied' in explanation

    def test_engine_comparison(self, compiler):
        """Test engine comparison."""
        text = "Implement microservices architecture"
        comparison = compiler.compare_engines(text)

        # Should have results for multiple engines
        assert len(comparison) >= 2
        
        for engine_name, result in comparison.items():
            if 'error' not in result:
                assert 'compression_ratio' in result
                assert 'quality_score' in result

    def test_cleanup(self, compiler):
        """Test proper cleanup."""
        # Should not raise any exceptions
        compiler.close()


class TestPatternManagement:
    """Test pattern management functionality."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database for testing."""
        with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
            db_path = f.name
        
        yield db_path
        
        if os.path.exists(db_path):
            os.unlink(db_path)

    @pytest.fixture
    def compiler(self, temp_db):
        """Create compiler for pattern testing."""
        config = CompilerConfig(
            database=DatabaseConfig(path=temp_db),
            log_level="ERROR"
        )
        return NeuralSemanticCompiler(config)

    def test_get_patterns(self, compiler):
        """Test getting patterns."""
        patterns = compiler.get_patterns()
        assert isinstance(patterns, list)

    def test_search_patterns(self, compiler):
        """Test pattern search."""
        # Add a test pattern first
        compiler.add_pattern("test pattern", "test", "word", "test")

        results = compiler.search_patterns("test")
        assert isinstance(results, list)

    def test_pattern_filtering(self, compiler):
        """Test pattern filtering by domain and type."""
        # Add patterns with different domains
        compiler.add_pattern("web test", "wt", "word", "web-development")
        compiler.add_pattern("agile test", "at", "word", "agile")

        web_patterns = compiler.get_patterns(domain="web-development")
        agile_patterns = compiler.get_patterns(domain="agile")

        # Should have different patterns
        web_originals = [p['original'] for p in web_patterns]
        agile_originals = [p['original'] for p in agile_patterns]

        if web_patterns:
            assert any("web" in orig.lower() for orig in web_originals)
        if agile_patterns:
            assert any("agile" in orig.lower() for orig in agile_originals)


class TestConfigurationManagement:
    """Test configuration management."""

    def test_default_config(self):
        """Test default configuration creation."""
        compiler = NeuralSemanticCompiler.create_default()
        assert compiler is not None

    def test_custom_config(self):
        """Test custom configuration."""
        config = CompilerConfig(
            compression={"default_level": CompressionLevel.AGGRESSIVE},
            log_level="DEBUG"
        )
        
        compiler = NeuralSemanticCompiler(config)
        assert compiler is not None


@pytest.mark.performance
class TestPerformance:
    """Performance tests."""

    @pytest.fixture
    def compiler(self):
        """Create compiler for performance testing."""
        return NeuralSemanticCompiler.create_default()

    def test_compression_speed(self, compiler):
        """Test compression speed."""
        text = "Build a production-ready React application with authentication and user management"

        import time
        start_time = time.time()
        result = compiler.compress(text)
        end_time = time.time()

        processing_time = (end_time - start_time) * 1000  # ms

        # Should complete in reasonable time
        assert processing_time < 1000  # Less than 1 second
        assert result.processing_time_ms < 500  # Internal timing should be reasonable

    def test_batch_compression(self, compiler):
        """Test batch compression performance."""
        texts = [
            "Build React application",
            "Implement user authentication", 
            "Create database schema",
            "Setup CI/CD pipeline",
            "Deploy to production"
        ]

        import time
        start_time = time.time()
        
        results = []
        for text in texts:
            result = compiler.compress(text)
            results.append(result)
        
        end_time = time.time()
        total_time = (end_time - start_time) * 1000

        # Should process all texts efficiently
        assert len(results) == len(texts)
        assert total_time < 5000  # Less than 5 seconds for 5 texts
        assert all(r.compression_ratio < 1.0 for r in results)


if __name__ == "__main__":
    pytest.main([__file__])