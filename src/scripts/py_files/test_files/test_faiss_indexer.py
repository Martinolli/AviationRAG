import pytest
import faiss
import numpy as np
import os
import json
import logging
from faiss_indexer import FAISSIndexer  # Import your FAISSIndexer class

# Create a temporary directory for test files
@pytest.fixture(scope="module")
def temp_dir(tmpdir_factory):
    return tmpdir_factory.mktemp("temp_faiss")

@pytest.fixture
def indexer(temp_dir, caplog):
    embedding_dim = 10
    with caplog.at_level(logging.INFO):
        indexer = FAISSIndexer(embedding_dim)
        assert "🛠️ Initializing FAISS index" in caplog.text
    return indexer

@pytest.fixture
def embeddings_and_metadata():
    embeddings = [
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.2, 0.3, 0.1, 0.5, 0.4, 0.2, 0.3, 0.1, 0.5, 0.4],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.2, 0.3, 0.1, 0.5, 0.4, 0.2, 0.3, 0.1, 0.5, 0.4],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5],
        [0.5, 0.4, 0.3, 0.2, 0.1, 0.5, 0.4, 0.3, 0.2, 0.1],
        [0.2, 0.3, 0.1, 0.5, 0.4, 0.2, 0.3, 0.1, 0.5, 0.4],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5],
    ]
    metadata = [
        {'chunk_id': 'chunk1', 'filename': 'file1.txt', 'text': 'text1', 'tokens': ['token1']},
        {'chunk_id': 'chunk2', 'filename': 'file2.txt', 'text': 'text2', 'tokens': ['token2']},
        {'chunk_id': 'chunk3', 'filename': 'file3.txt', 'text': 'text3', 'tokens': ['token3']},
        {'chunk_id': 'chunk4', 'filename': 'file4.txt', 'text': 'text4', 'tokens': ['token4']},
        {'chunk_id': 'chunk5', 'filename': 'file5.txt', 'text': 'text5', 'tokens': ['token5']},
        {'chunk_id': 'chunk6', 'filename': 'file6.txt', 'text': 'text6', 'tokens': ['token6']},
        {'chunk_id': 'chunk7', 'filename': 'file7.txt', 'text': 'text7', 'tokens': ['token7']},
        {'chunk_id': 'chunk8', 'filename': 'file8.txt', 'text': 'text8', 'tokens': ['token8']},
        {'chunk_id': 'chunk9', 'filename': 'file9.txt', 'text': 'text9', 'tokens': ['token9']},
        {'chunk_id': 'chunk10', 'filename': 'file10.txt', 'text': 'text10', 'tokens': ['token10']},
    ]
    return embeddings, metadata

def test_index_creation(indexer, caplog):
    #The index creation is now in the fixture
    assert isinstance(indexer.index, faiss.IndexIVFFlat)

def test_train_index(indexer, embeddings_and_metadata, caplog):
    embeddings, _ = embeddings_and_metadata
    with caplog.at_level(logging.INFO):
        indexer.train(embeddings)
        assert indexer.index.is_trained
        assert indexer.is_trained
        assert "🏋️‍♀️ Training FAISS index" in caplog.text
        assert "✅ FAISS index trained" in caplog.text

def test_add_embeddings(indexer, embeddings_and_metadata, caplog):
    embeddings, metadata = embeddings_and_metadata
    indexer.train(embeddings)
    with caplog.at_level(logging.INFO):
        indexer.add_embeddings(embeddings, metadata)
        assert indexer.index.ntotal == len(embeddings)
        assert len(indexer.metadata) == len(metadata)
        assert "➕ Adding" in caplog.text
        assert "✅ Added" in caplog.text

def test_search(indexer, embeddings_and_metadata, caplog):
    embeddings, metadata = embeddings_and_metadata
    indexer.train(embeddings)
    indexer.add_embeddings(embeddings, metadata)
    query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5]
    with caplog.at_level(logging.INFO):
        results = indexer.search(query_embedding, k=2)
        assert len(results) == 2
        assert isinstance(results[0][0], dict)
        assert 'chunk_id' in results[0][0]
        assert "🔍 Searching FAISS index" in caplog.text
        assert "✅ Search completed" in caplog.text

def test_save_and_load(indexer, embeddings_and_metadata, temp_dir, caplog):
    embeddings, metadata = embeddings_and_metadata
    indexer.train(embeddings)
    indexer.add_embeddings(embeddings, metadata)
    
    save_path = os.path.join(temp_dir, "test_index")
    with caplog.at_level(logging.INFO):
        indexer.save(save_path)
        assert "💾 Saving FAISS index" in caplog.text
        assert "✅ FAISS index saved" in caplog.text

    with caplog.at_level(logging.INFO):
        loaded_indexer = FAISSIndexer.load(save_path)
        assert loaded_indexer.index.ntotal == len(embeddings)
        assert len(loaded_indexer.metadata) == len(metadata)
        assert loaded_indexer.is_trained
        assert "📂 Loading FAISS index" in caplog.text
        assert "✅ FAISS index loaded" in caplog.text

def test_load_from_file(indexer, embeddings_and_metadata, temp_dir, caplog):
    embeddings, metadata = embeddings_and_metadata
    
    # Create a dummy JSON file for testing
    file_path = os.path.join(temp_dir, "test_embeddings.json")
    data = [{"embedding": emb, **meta} for emb, meta in zip(embeddings, metadata)]
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f)

    with caplog.at_level(logging.INFO):
        loaded_indexer = FAISSIndexer.load_from_file(file_path)
        assert loaded_indexer.index.ntotal == len(embeddings)
        assert len(loaded_indexer.metadata) == len(metadata)
        assert loaded_indexer.is_trained
        assert "📂 Loading embeddings from file" in caplog.text
        assert "📚 Loaded" in caplog.text

def test_add_embeddings_before_train(indexer, embeddings_and_metadata):
    embeddings, metadata = embeddings_and_metadata
    with pytest.raises(RuntimeError):
        indexer.add_embeddings(embeddings, metadata)

def test_search_before_train(indexer, embeddings_and_metadata):
    query_embedding = [0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.2, 0.3, 0.4, 0.5]
    with pytest.raises(RuntimeError):
        indexer.search(query_embedding, k=2)
