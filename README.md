# Neural Semantic Compiler

[![License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](LICENSE)
[![Copyright](https://img.shields.io/badge/Copyright-Brenner%20Cruvinel-blue.svg)](https://github.com/brennercruvinel)
[![Version](https://img.shields.io/badge/version-0.3.0--alpha-orange.svg)](https://github.com/brennercruvinel/neuralsemantic)
[![Status](https://img.shields.io/badge/status-early%20development-orange.svg)](https://github.com/brennercruvinel/neuralsemantic)

> **⚠️ PROPRIETARY SOFTWARE - ALL RIGHTS RESERVED ⚠️**
> 
> Copyright © 2025 Brenner Cruvinel. This software contains proprietary algorithms and trade secrets. Unauthorized use, reproduction, or distribution is strictly prohibited.
>
> **IMPORTANT: This project is currently in early development (approximately 20% complete). The application is not production-ready and contains bugs. We are actively implementing core features and comprehensive testing.**
[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Node](https://img.shields.io/badge/node-16+-green.svg)](https://nodejs.org/)
[![NPM](https://img.shields.io/badge/npm-%40neurosemantic-red.svg)](https://www.npmjs.com/org/neurosemantic)

## Project Status

**Current State**: Early Alpha (v0.3.0)
- **Completion**: ~20%
- **Core Features**: Under active development
- **Testing**: Implementing comprehensive test suite
- **Stability**: Expect bugs and breaking changes
- **API**: Subject to significant changes

## Overview

Neural Semantic Compiler (NSC) is a semantic compression tool that reduces LLM token usage by 40-65% while preserving complete semantic meaning with, not just reduce characters, it's revesability engeerning. It works as a preprocessing layer for any LLM interaction, making AI more accessible and cost-effective.

**Note**: The examples and features described below represent the project's vision. Many are not yet implemented or may work partially.

## Installation

**⚠️ Warning**: Package not yet published to PyPI/NPM. Installation from source only.

**Python (from source):**
```bash
git clone https://github.com/brennercruvinel/neuralsemantic.git
cd neuralsemantic/python
pip install -e .
```

**Node.js (coming soon):**
```bash
# Not yet available
# npm install @neurosemantic/core @neurosemantic/cli
```

**From Source:**
```bash
git clone https://github.com/brennercruvinel/neuralsemantic.git
cd neuralsemantic
pip install -e .
```

## Quick Start

⚠️ **Note**: Most features shown below are planned but not yet implemented.

### Command Line (Limited Functionality)

```bash
# Basic compression (partially working)
python -m neuralsemantic.cli.main compress "Your text here"

# These commands are planned but NOT YET IMPLEMENTED:
# echo "Build a production-ready React application with authentication" | nsc compress
# nsc compress --domain web-dev "Implement user authentication system"
# nsc compress --show-stats "Create REST API with Express.js"
```

### Python

```python
from neuralsemantic import NeuralSemanticCompiler

# Initialize
compiler = NeuralSemanticCompiler()

# Compress text
result = compiler.compress("Build a production-ready React application")

print(f"Original: {result.original_text}")
print(f"Compressed: {result.compressed_text}")
print(f"Token reduction: {(1-result.compression_ratio):.1%}")
```

### Node.js

```javascript
import { NeuralSemanticCompiler } from '@neurosemantic/core';

const compiler = new NeuralSemanticCompiler();
const result = await compiler.compress("Build a production-ready React application");

console.log(`Compressed: ${result.compressedText}`);
console.log(`Token reduction: ${(1 - result.compressionRatio) * 100}%`);
```

## Examples

### Web Development
```
Input:  "Build a production-ready React application with user authentication,
         real-time data synchronization, responsive design, error handling"

Output: "bld prod-rdy React app w/ usr auth, RT sync, rsp design, err hdl"

Result: 53% reduction (12 tokens → 5 tokens)
```

### Technical Architecture
```
Input:  "Implement microservices architecture with Docker containerization,
         Kubernetes orchestration, API gateway, monitoring"

Output: "impl μsvc arch w/ Docker, k8s orch, API gtw, mon"

Result: 44% reduction (14 tokens → 8 tokens)
```

## Features

### Core Capabilities

- **Token Reduction**: 40-65% average compression ratio
- **Semantic Preservation**: Maintains 100% of meaning
- **Real-time Processing**: < 20ms average latency
- **Framework Agnostic**: Works with any LLM (GPT-4, Claude, Llama)
- **Domain Optimization**: Specialized patterns for different domains

### Supported Domains

- `general` - Default compression patterns
- `web-dev` - Web development and frontend
- `data-science` - Data analysis and ML
- `devops` - Infrastructure and deployment
- `agile` - Project management

## Configuration

Create `~/.neuralsemantic/config.json`:

```json
{
  "compression": {
    "default_level": "balanced",
    "semantic_threshold": 0.90,
    "preserve_code": true,
    "preserve_urls": true
  },
  "domains": ["general", "web-dev", "data-science"],
  "vector": {
    "model": "sentence-transformers/all-MiniLM-L6-v2"
  }
}
```

### Compression Levels

- `light` - Conservative compression, highest quality
- `balanced` - Optimal balance (default)
- `aggressive` - Maximum compression

## Advanced Usage

### Pattern Learning

Learn patterns from your codebase:

```bash
nsc learn --corpus-path ./src --min-frequency 3
```

### Custom Patterns

Add domain-specific patterns:

```bash
nsc add-pattern "authentication system" "auth sys" --domain web-dev
```

### Batch Processing

```python
documents = load_documents("./docs")
results = compiler.batch_compress(documents)

for result in results:
    print(f"Saved {result.token_savings} tokens")
```

## API Reference

### Compression Result

```python
CompressionResult:
    original_text: str          # Original input text
    compressed_text: str        # Compressed output
    compression_ratio: float    # Compression ratio (0-1)
    quality_score: float        # Quality score (0-10)
    token_savings: int          # Tokens saved
    processing_time_ms: int     # Processing time
```

### Core Methods

```python
# Basic compression
result = compiler.compress(text, level="balanced", domain="general")

# Batch processing
results = compiler.batch_compress(documents)

# Pattern management
compiler.add_pattern(original, compressed, domain, priority)
compiler.learn_from_corpus(documents)
```

## Architecture

NSC uses a hybrid architecture combining:

1. **Pattern Matching** - SQLite database for structured patterns
2. **Vector Similarity** - ChromaDB for semantic search
3. **ML Optimization** - Automatic pattern discovery
4. **Quality Validation** - Ensures semantic preservation

### Processing Pipeline

1. Semantic analysis and concept extraction
2. Domain-specific pattern matching
3. Vector similarity search
4. Compression with quality validation
5. Smart fallback for edge cases

## Performance

| Metric | Value |
|--------|-------|
| Average Compression | 40-65% |
| Processing Speed | < 20ms |
| Semantic Preservation | > 95% |
| Memory Usage | < 100MB |
| Throughput | 10,000 req/s |

## Integration

### Supported LLMs
- OpenAI 
- Anthropic Claude
- China models
- Google Gemini
- Meta Llama
- Mistral AI
- Local models (Ollama)

### Framework Integration
- LangChain
- LlamaIndex
- Semantic Kernel
- AutoGPT

## Development

### Setup

```bash
git clone https://github.com/brennercruvinel/neuralsemantic.git
cd neuralsemantic
pip install -e .[dev]
```

### Testing

```bash
# Run tests
pytest

# Run with coverage
pytest --cov=neuralsemantic

# Run benchmarks
pytest tests/benchmarks/
```

### Development Roadmap

### Implemented (✓)
- Basic project structure
- Core compression algorithms (partial)
- Simple pattern matching
- Basic CLI interface

### In Progress ()
- Comprehensive test suite
- Pattern database implementation
- Vector similarity search
- Quality validation system
- Domain-specific patterns

### Planned ()
- ChromaDB integration
- ML-based pattern learning
- Multi-language support
- API server
- Performance optimizations
- Documentation
- Package publishing (PyPI/NPM)

## Known Issues

- Pattern matching is inconsistent
- No proper error handling
- Memory leaks in vector operations
- CLI commands may fail unexpectedly
- Compression quality varies significantly
- No data persistence between sessions

## Contributing

**This is proprietary software. External contributions are not accepted.**

## License

**PROPRIETARY LICENSE** - All Rights Reserved

This software is protected by copyright and contains proprietary algorithms and trade secrets. See [LICENSE](LICENSE) for details.

## Contact & Licensing

For licensing inquiries or commercial use:

- **Author**: Brenner Cruvinel
- **Email**: cruvinelbrenner@gmail.com
- **GitHub**: [@brennercruvinel](https://github.com/brennercruvinel)

## Legal Notice

Copyright © 2025 Brenner Cruvinel. All Rights Reserved.

Unauthorized use, reproduction, or distribution of this software is strictly prohibited and will be prosecuted to the fullest extent of the law.
