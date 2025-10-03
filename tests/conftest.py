"""
Pytest configuration and shared fixtures for FinRAG tests.
Includes mocks for expensive operations like embeddings.
"""
import pytest
from unittest.mock import Mock, patch
import numpy as np


@pytest.fixture
def mock_embeddings():
    """Mock HuggingFaceEmbeddings to avoid downloading models during tests"""
    with patch('langchain_community.embeddings.HuggingFaceEmbeddings') as mock:
        # Create a mock that returns consistent fake embeddings
        instance = Mock()
        instance.embed_documents = Mock(side_effect=lambda texts: [
            np.random.rand(384).tolist() for _ in texts
        ])
        instance.embed_query = Mock(side_effect=lambda text: np.random.rand(384).tolist())
        mock.return_value = instance
        yield mock


@pytest.fixture
def mock_llm():
    """Mock HuggingFace LLM to avoid model loading during tests"""
    with patch('transformers.pipeline') as mock_pipeline:
        mock_pipe = Mock()
        mock_pipe.return_value = [{"generated_text": "This is a mocked answer."}]
        mock_pipeline.return_value = mock_pipe
        yield mock_pipeline
