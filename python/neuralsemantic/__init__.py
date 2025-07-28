"""
Neural Semantic Compiler - The first compiler for neural communication

Copyright (c) 2024 Brenner Cruvinel (@brennercruvinel)
All Rights Reserved.

PROPRIETARY AND CONFIDENTIAL
This software contains proprietary algorithms and trade secrets.
Unauthorized copying, reverse engineering, or distribution is strictly prohibited.

For licensing inquiries: cruvinelbrenner@gmail.com
"""

from .core.compiler import NeuralSemanticCompiler
from .core.types import CompressionResult, CompressionLevel, CompressionContext
from .core.exceptions import CompressionError, PatternConflictError, QualityError
from .core.config import CompilerConfig, DatabaseConfig, VectorConfig, CompressionConfig

__version__ = "1.0.0"
__author__ = "Brenner Cruvinel"
__email__ = "cruvinelbrenner@gmail.com"
__homepage__ = "https://neurosemantic.tech"

__all__ = [
    "NeuralSemanticCompiler",
    "CompressionResult", 
    "CompressionLevel",
    "CompressionContext",
    "CompressionError",
    "PatternConflictError", 
    "QualityError",
    "CompilerConfig",
    "DatabaseConfig",
    "VectorConfig", 
    "CompressionConfig",
]