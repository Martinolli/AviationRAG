import pytest
import os
import logging
from unittest.mock import MagicMock, patch
from AviationRAG.src.scripts.py_files.test_files.test_aviationai_helper import get_embedding, clear_cache, embedding_cache
from AviationRAG.src.scripts.py_files.aviationai import client # Import client from aviationai_1.py

# Mock the OpenAI client for testing
@pytest.fixture() # Removed autouse=True
def mock_openai():
    with patch.object(client.embeddings, "create") as mock_create: # Patch the client from aviationai_1.py
        mock_create.return_value.data = [
            MagicMock(embedding=[0.1, 0.2, 0.3])
        ]
        yield mock_create

@pytest.mark.usefixtures("mock_openai") # Added this line
def test_get_embedding_cache_hit(mock_openai):
    # First call should generate a new embedding
    clear_cache()
    embedding1 = get_embedding("test query")
    assert mock_openai.call_count == 1
    assert "test query" in embedding_cache # Now it is imported
    assert embedding_cache["test query"] == [0.1, 0.2, 0.3] # Now it is imported

    # Second call with the same query should use the cached embedding
    embedding2 = get_embedding("test query")
    assert mock_openai.call_count == 1  # No new API call
    assert embedding1 == embedding2

@pytest.mark.usefixtures("mock_openai") # Added this line
def test_get_embedding_cache_miss(mock_openai):
    # First call should generate a new embedding
    clear_cache()
    embedding1 = get_embedding("test query 1")
    assert mock_openai.call_count == 1
    assert "test query 1" in embedding_cache # Now it is imported

    # Second call with a different query should generate a new embedding
    embedding2 = get_embedding("test query 2")
    assert mock_openai.call_count == 2
    assert "test query 2" in embedding_cache # Now it is imported

@pytest.mark.usefixtures("mock_openai") # Added this line
def test_clear_embedding_cache(mock_openai):
    # Add some embeddings to the cache
    clear_cache()
    get_embedding("test query 1")
    get_embedding("test query 2")
    assert len(embedding_cache) == 2 # Now it is imported

    # Clear the cache
    clear_cache()
    assert len(embedding_cache) == 0 # Now it is imported

@pytest.mark.usefixtures("mock_openai") # Added this line
def test_get_embedding_after_clear_cache(mock_openai):
    # Add some embeddings to the cache
    clear_cache()
    get_embedding("test query 1")
    assert mock_openai.call_count == 1
    assert len(embedding_cache) == 1 # Now it is imported

    # Clear the cache
    clear_cache()
    assert len(embedding_cache) == 0 # Now it is imported

    # Call get_embedding again after clearing the cache
    get_embedding("test query 1")
    assert mock_openai.call_count == 2
    assert len(embedding_cache) == 1 # Now it is imported
