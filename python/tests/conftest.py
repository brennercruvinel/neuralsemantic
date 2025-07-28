"""
Configuração global de testes para Neural Semantic Compiler.
"""
import os
import sys
from unittest.mock import MagicMock

# Set test mode
os.environ['NSC_TEST_MODE'] = '1'

# Mock heavy dependencies before they're imported
sys.modules['torch'] = MagicMock()
sys.modules['sentence_transformers'] = MagicMock()
sys.modules['chromadb'] = MagicMock()

# Mock specific classes
class MockSentenceTransformer:
    def __init__(self, model_name):
        self.model_name = model_name
    
    def encode(self, texts, **kwargs):
        # Return mock embeddings
        if isinstance(texts, str):
            return [0.1] * 384
        return [[0.1] * 384 for _ in texts]

# Replace with mock
sys.modules['sentence_transformers'].SentenceTransformer = MockSentenceTransformer