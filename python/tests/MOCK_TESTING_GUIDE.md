# Mock Testing Guide for Neural Semantic Compiler

This guide provides comprehensive examples of mock testing strategies for incremental development of the Neural Semantic Compiler. It shows how to create effective mocks, test interfaces before implementation, and gradually replace mocks with real implementations.

## Overview

Mock testing is essential for:
- **Incremental Development**: Build and test components independently
- **External Dependencies**: Test without requiring ChromaDB, vector stores, or ML models
- **Fast Iteration**: Quick feedback loops during development
- **Isolation**: Test individual components without side effects
- **Edge Cases**: Simulate error conditions and unusual scenarios

## File Structure

```
tests/
├── test_mock_examples_python.py      # Python/pytest mock examples
├── test_mock_examples_typescript.ts   # TypeScript/Jest mock examples  
├── gradual_implementation_guide.py   # Step-by-step transition guide
├── practical_mock_implementations.py # Ready-to-use mock implementations
└── MOCK_TESTING_GUIDE.md            # This documentation
```

## 1. Python Mock Testing Examples

### Basic Mock Setup

```python
from unittest.mock import Mock, patch
import pytest

# Mock external dependencies
@pytest.fixture
def mock_chromadb_client():
    mock_client = Mock()
    mock_collection = Mock()
    
    mock_collection.query.return_value = {
        "documents": [["sample pattern", "another pattern"]],
        "distances": [[0.1, 0.3]],
        "metadatas": [[{"type": "compound"}, {"type": "word"}]]
    }
    
    mock_client.get_or_create_collection.return_value = mock_collection
    return mock_client

# Test with mocked dependencies
@patch('neuralsemantic.vector.vector_store.chromadb')
def test_compiler_with_mocked_vector_store(mock_chromadb):
    mock_chromadb.Client.return_value = mock_chromadb_client()
    
    from neuralsemantic.core.compiler import NeuralSemanticCompiler
    compiler = NeuralSemanticCompiler(config)
    
    assert compiler is not None
    mock_chromadb.Client.assert_called()
```

### Mocking Database Operations

```python
def test_pattern_manager_with_mock_database():
    with patch('sqlite3.connect') as mock_connect:
        mock_cursor = Mock()
        mock_connection = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection
        
        # Mock database responses
        mock_cursor.fetchall.return_value = [
            (1, "machine learning", "ML", "compound", "ai", 800, 1.0),
            (2, "user interface", "UI", "compound", "web", 750, 0.9)
        ]
        
        from neuralsemantic.patterns.pattern_manager import PatternManager
        manager = PatternManager("mock_db.db")
        patterns = manager.get_patterns()
        
        assert len(patterns) >= 0
        mock_connect.assert_called_with("mock_db.db")
```

### Async Operation Mocking

```python
@pytest.mark.asyncio
async def test_async_compression_pipeline():
    mock_compressor = AsyncMock()
    mock_compressor.compress_async.return_value = MockCompressionResult(
        original_text="Async compression test",
        compressed_text="Async compress test",
        compression_ratio=0.8,
        quality_score=0.9,
        # ... other fields
    )
    
    result = await mock_compressor.compress_async("Async compression test")
    assert result.compression_ratio < 1.0
```

## 2. TypeScript/Jest Mock Examples

### Basic Jest Mocking

```typescript
import { jest, describe, it, expect } from '@jest/globals';

describe('Vector Store Mocking', () => {
  let mockChromaClient: jest.Mocked<any>;
  let mockCollection: jest.Mocked<any>;
  
  beforeEach(() => {
    mockCollection = {
      add: jest.fn().mockResolvedValue(undefined),
      query: jest.fn().mockResolvedValue({
        documents: [['sample pattern', 'another pattern']],
        distances: [[0.1, 0.3]],
        metadatas: [[
          { type: 'compound', domain: 'web' },
          { type: 'word', domain: 'ai' }
        ]]
      }),
      count: jest.fn().mockResolvedValue(100)
    };
    
    mockChromaClient = {
      getOrCreateCollection: jest.fn().mockResolvedValue(mockCollection)
    };
  });
  
  it('should mock vector store operations', async () => {
    const vectorStore = new MockVectorStore(mockChromaClient);
    
    const addResult = await vectorStore.addPattern('test pattern', 'test');
    expect(addResult).toBe(true);
    
    const searchResults = await vectorStore.searchSimilar('machine learning');
    expect(searchResults).toHaveLength(2);
  });
});
```

### Complex Side Effects

```typescript
it('should use side effects for complex mock behaviors', () => {
  const mockCompressor = {
    compress: jest.fn()
  };
  
  mockCompressor.compress.mockImplementation((text: string, level?: string) => {
    if (text.length < 10) {
      return {
        originalText: text,
        compressedText: text, // No compression for short text
        compressionRatio: 1.0,
        qualityScore: 1.0
      };
    }
    
    return {
      originalText: text,
      compressedText: text.substring(0, Math.floor(text.length / 2)),
      compressionRatio: 0.5,
      qualityScore: 0.8
    };
  });
  
  const shortResult = mockCompressor.compress('Hi');
  expect(shortResult.compressionRatio).toBe(1.0);
  
  const longResult = mockCompressor.compress('This is a much longer text');
  expect(longResult.compressionRatio).toBe(0.5);
});
```

## 3. Gradual Implementation Strategy

### Phase 1: Full Mocking (Weeks 1-2)

**Goal**: Define interfaces and test contracts

```python
class CompressionEngine(ABC):
    @abstractmethod
    def compress(self, text: str, level: str = "balanced") -> Dict[str, Any]:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass

def test_compression_engine_interface():
    mock_engine = Mock(spec=CompressionEngine)
    mock_engine.compress.return_value = {
        "original_text": "Build a React application",
        "compressed_text": "Build React app",
        "compression_ratio": 0.75,
        "quality_score": 0.9
    }
    
    # Test interface contract
    result = mock_engine.compress("Build a React application")
    assert result["compression_ratio"] < 1.0
```

**Checklist**:
- ✓ Define abstract interfaces for all components
- ✓ Create comprehensive mock implementations
- ✓ Write tests that verify interface contracts
- ✓ Document expected behaviors and return types
- ✓ Test error conditions with mocks

### Phase 2: Partial Implementation (Weeks 3-4)

**Goal**: Implement business logic with mocked dependencies

```python
class PartialSemanticEngine:
    def __init__(self, vector_store):
        self.vector_store = vector_store  # Mocked dependency
    
    def compress(self, text: str, level: str = "balanced") -> Dict[str, Any]:
        # Real compression logic using mocked vector store
        words = text.split()
        compressed_words = []
        
        for word in words:
            similar_patterns = self.vector_store.search_similar(word)
            if similar_patterns and similar_patterns[0]["similarity"] > 0.9:
                compressed_words.append(similar_patterns[0]["compressed"])
            else:
                compressed_words.append(word)
        
        return {
            "original_text": text,
            "compressed_text": " ".join(compressed_words),
            "compression_ratio": len(" ".join(compressed_words)) / len(text)
        }
```

**Checklist**:
- ✓ Implement core business logic
- ✓ Mock external dependencies (database, APIs, file system)
- ✓ Test business logic thoroughly
- ✓ Ensure proper error handling
- ✓ Validate input/output transformations

### Phase 3: Real Implementation (Weeks 5-6)

**Goal**: Use real components in controlled environment

```python
@pytest.fixture
def temp_database():
    with tempfile.NamedTemporaryFile(suffix='.db', delete=False) as f:
        db_path = f.name
    
    # Initialize real database schema
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE patterns (...)""")
    conn.commit()
    conn.close()
    
    yield db_path
    
    if os.path.exists(db_path):
        os.unlink(db_path)

def test_real_pattern_manager_with_temp_database(temp_database):
    manager = RealPatternManager(temp_database)  # Real implementation
    
    assert manager.add_pattern("machine learning", "ML", "compound", "ai")
    patterns = manager.get_patterns()
    assert len(patterns) > 0
```

**Checklist**:
- ✓ Replace mocks with real implementations one by one
- ✓ Use temporary/test databases and file systems
- ✓ Mock only expensive external services
- ✓ Test with realistic data volumes
- ✓ Verify performance characteristics

### Phase 4: Integration Testing (Weeks 7-8)

**Goal**: End-to-end testing with minimal mocking

```python
def test_full_compression_pipeline():
    # Only mock external services you don't control
    with patch('requests.post') as mock_external_api:
        mock_external_api.return_value.json.return_value = {"status": "success"}
        
        # Use real compiler with real database
        compiler = NeuralSemanticCompiler(real_config)
        result = compiler.compress("Build a production-ready React application")
        
        assert result.engine_used in ["SemanticEngine", "HybridEngine"]
        mock_external_api.assert_not_called()  # Verify external API wasn't needed
```

**Checklist**:
- ✓ Test complete workflows end-to-end
- ✓ Use real data and realistic scenarios
- ✓ Mock only external services you don't control
- ✓ Test performance under load
- ✓ Verify system behavior under failures

## 4. Component-Specific Mock Strategies

### ChromaDB Vector Store

```python
@pytest.fixture
def mock_chromadb_client():
    mock_client = Mock()
    mock_collection = Mock()
    
    # Mock collection methods
    mock_collection.add.return_value = None
    mock_collection.query.return_value = {
        "documents": [["pattern1", "pattern2"]],
        "distances": [[0.1, 0.3]],
        "metadatas": [[{"type": "compound"}, {"type": "word"}]]
    }
    
    mock_client.get_or_create_collection.return_value = mock_collection
    return mock_client

# Use in tests
@patch('neuralsemantic.vector.vector_store.chromadb')
def test_with_mocked_chromadb(mock_chromadb):
    mock_chromadb.Client.return_value = mock_chromadb_client()
    # Test your code here
```

### Sentence Transformers

```python
@pytest.fixture
def mock_sentence_transformer():
    mock_transformer = Mock()
    # Return realistic 384-dimensional embeddings
    mock_transformer.encode.return_value = [[0.1] * 384] * 2
    return mock_transformer

@patch('neuralsemantic.vector.embeddings.SentenceTransformer')
def test_with_mocked_embeddings(mock_transformer_class):
    mock_transformer_class.return_value = mock_sentence_transformer()
    # Test embedding-dependent code
```

### Database Operations

```python
def test_with_mocked_database():
    with patch('sqlite3.connect') as mock_connect:
        mock_cursor = Mock()
        mock_connection = Mock()
        mock_connection.cursor.return_value = mock_cursor
        mock_connect.return_value = mock_connection
        
        # Mock specific queries
        mock_cursor.fetchall.return_value = [
            (1, "pattern", "P", "word", "general", 500, 1.0)
        ]
        
        # Test your database-dependent code
```

## 5. Error Handling and Edge Cases

### Simulating Failures

```python
def test_database_connection_failure():
    with patch('sqlite3.connect') as mock_connect:
        mock_connect.side_effect = Exception("Database unavailable")
        
        with pytest.raises(Exception) as exc_info:
            manager = PatternManager("failing_db.db")
        
        assert "Database unavailable" in str(exc_info.value)

def test_vector_store_unavailable():
    with patch('neuralsemantic.vector.vector_store.CHROMADB_AVAILABLE', False):
        compiler = NeuralSemanticCompiler(config)
        result = compiler.compress("test")
        
        # Should work with degraded functionality
        assert result.engine_used == "FallbackEngine"
```

### Graceful Degradation

```python
def test_graceful_degradation():
    compiler = MockNeuralSemanticCompiler({
        'vector_store_available': False,
        'database_available': True
    })
    
    result = compiler.compress("Test text")
    health = compiler.validate_system_health()
    
    assert result.compression_ratio <= 1.0  # Still works
    assert health.overall_status == "warning"  # But with warnings
```

## 6. Performance Testing with Mocks

### Mock Performance Scenarios

```python
def test_compression_speed():
    mock_compiler = Mock()
    
    def mock_compress(text):
        # Simulate processing time
        time.sleep(0.001)  # 1ms
        return MockCompressionResult(
            processing_time_ms=1.0,
            # ... other fields
        )
    
    mock_compiler.compress.side_effect = mock_compress
    
    start_time = time.time()
    result = mock_compiler.compress("test text")
    end_time = time.time()
    
    assert (end_time - start_time) * 1000 < 10  # Less than 10ms
    assert result.processing_time_ms < 5.0
```

### Batch Operation Testing

```python
@pytest.mark.asyncio
async def test_batch_performance():
    mock_compiler = MockAsyncCompiler()
    
    texts = ["Text " + str(i) for i in range(100)]
    
    start_time = time.time()
    results = await mock_compiler.compress_batch_async(texts)
    end_time = time.time()
    
    assert len(results) == 100
    assert (end_time - start_time) < 1.0  # Less than 1 second for 100 items
```

## 7. Best Practices

### Mock Design Principles

1. **Mock Interfaces, Not Implementations**
   ```python
   # Good: Mock the interface
   mock_engine = Mock(spec=CompressionEngine)
   
   # Bad: Mock implementation details
   mock_engine = Mock()
   mock_engine._internal_method = Mock()
   ```

2. **Use Realistic Data**
   ```python
   # Good: Realistic mock data
   mock_pattern = {
       "original": "machine learning",
       "compressed": "ML",
       "type": "compound",
       "quality": 0.95
   }
   
   # Bad: Unrealistic mock data
   mock_pattern = {"x": "y", "z": 999}
   ```

3. **Test Behavior, Not Implementation**
   ```python
   # Good: Test behavior
   result = compiler.compress("test")
   assert result.compression_ratio < 1.0
   
   # Bad: Test implementation details
   assert compiler._internal_counter == 1
   ```

### Mock Maintenance

1. **Keep Mocks Simple**: Only mock what you need for the test
2. **Update Mocks with Interfaces**: When interfaces change, update mocks
3. **Use Factories**: Create mock factories for complex objects
4. **Document Mock Behavior**: Explain what each mock simulates

```python
class MockFactory:
    @staticmethod
    def create_compression_result(compression_ratio=0.8, quality=0.9):
        """Create a realistic compression result for testing."""
        return MockCompressionResult(
            original_text="test input",
            compressed_text="test output",
            compression_ratio=compression_ratio,
            quality_score=quality,
            # ... other realistic defaults
        )
```

## 8. Integration with CI/CD

### Test Configuration

```python
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers = [
    "unit: unit tests with mocks",
    "integration: integration tests with real components", 
    "performance: performance tests",
    "mock: tests using mock implementations"
]

# Run different test suites
# pytest -m "unit"           # Only mock-based unit tests
# pytest -m "integration"    # Only integration tests
# pytest -m "not slow"       # Skip slow tests in CI
```

### Environment-Specific Testing

```python
# Use environment variables to control mocking
import os

def should_use_real_vector_store():
    return os.getenv('USE_REAL_VECTOR_STORE', 'false').lower() == 'true'

@pytest.fixture
def vector_store():
    if should_use_real_vector_store():
        return RealVectorStore()
    else:
        return MockVectorStore()
```

## 9. Common Pitfalls and Solutions

### Pitfall 1: Over-Mocking
```python
# Bad: Mocking too much
with patch('os.path.exists'), patch('open'), patch('json.loads'):
    # Test becomes meaningless
    
# Good: Mock only external dependencies  
with patch('requests.get'):
    # Test your actual logic
```

### Pitfall 2: Brittle Mocks
```python
# Bad: Brittle mock tied to implementation
mock_obj.method.assert_called_with(exact_internal_parameter)

# Good: Test behavior, not implementation
assert result.status == "success"
```

### Pitfall 3: Unrealistic Mocks
```python
# Bad: Unrealistic mock behavior
mock_api.get.return_value = "success"  # APIs don't return strings

# Good: Realistic mock behavior
mock_api.get.return_value = Mock(status_code=200, json=lambda: {"status": "success"})
```

## 10. Ready-to-Use Mock Templates

The `practical_mock_implementations.py` file provides complete, ready-to-use mock implementations:

- `MockNeuralSemanticCompiler`: Full compiler with realistic behavior
- `MockVectorStore`: Vector similarity operations
- `MockPatternManager`: Database pattern management
- `MockQualityScorer`: Compression quality evaluation
- `MockAsyncCompiler`: Async operations
- `MockTestEnvironment`: Complete test environment setup

### Quick Start

```python
from practical_mock_implementations import MockNeuralSemanticCompiler

# Create a fully functional mock compiler
compiler = MockNeuralSemanticCompiler.create_default()

# Use it like the real thing
result = compiler.compress("Build a React application")
print(f"Compressed: {result.compressed_text}")
print(f"Quality: {result.quality_score}")

# Get comprehensive statistics
stats = compiler.get_statistics()
print(f"Compressions: {stats['session']['compressions']}")
```

## Conclusion

This mock testing strategy enables:

1. **Fast Development**: Build and test components independently
2. **Reliable Tests**: Consistent, reproducible test results
3. **Easy Debugging**: Isolate issues to specific components
4. **Incremental Progress**: Gradually replace mocks with real implementations
5. **Comprehensive Coverage**: Test edge cases and error conditions

Follow the four-phase approach to systematically transition from full mocks to complete integration, ensuring robust and maintainable code throughout the development process.