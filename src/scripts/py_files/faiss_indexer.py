"""
FAISS Indexer module for managing vector embeddings and similarity search.

This module provides the FAISSIndexer class for training, adding, and searching
embeddings using Facebook AI Similarity Search (FAISS) indexes.
"""

import json
import logging
import faiss
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class FAISSIndexer:
    """
    Class for managing a FAISS index, including training, adding embeddings, and searching.
    Attributes:
        embedding_dim (int): The dimension of the embedding vectors.
        nlist (int): The number of clusters for the IVF index.
        quantizer (faiss.IndexFlatL2): The quantizer used for the IVF index.
        index (faiss.IndexIVFFlat): The FAISS index for storing embeddings.
        metadata (dict): A dictionary mapping index positions to metadata.
        is_trained (bool): Flag indicating whether the index has been trained.
    """
    def __init__(
        self, embedding_dim, nlist=10, verbose=False
    ):  # Reduced nlist for smaller datasets
        self.verbose = verbose
        if self.verbose:
            logging.info(
                "🛠️ Initializing FAISS index with embedding dimension: %s, nlist: %s",
                embedding_dim,
                nlist
            )
        self.embedding_dim = embedding_dim
        self.nlist = nlist
        self.quantizer = faiss.IndexFlatL2(embedding_dim)
        self.index = faiss.IndexIVFFlat(
            self.quantizer, embedding_dim, nlist, faiss.METRIC_L2
        )
        self.metadata = {}
        self.is_trained = False

    def train(self, embeddings):
        """Train the index with a set of embeddings.
                Args:
            embeddings (list of list or np.array): The embedding vectors to train the index with.
        Returns:
            None
        """
        logging.info("🏋️‍♀️ Training FAISS index with %d embeddings...", len(embeddings))
        embeddings_array = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings_array)
        self.index.train(embeddings_array)
        self.is_trained = True
        logging.info("✅ FAISS index trained with %d embeddings.", len(embeddings))

    def add_embeddings(self, embeddings, metadata):
        """
        Add embeddings and their corresponding metadata to the index.
        Args:
            embeddings (list of list or np.array): The embedding vectors to add.
            metadata (list of dict): The metadata associated with each embedding.
        Returns:     None
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding embeddings.")
        logging.info("➕ Adding %d embeddings to the FAISS index...", len(embeddings))
        embeddings_array = np.array(embeddings).astype("float32")
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)
        for i, meta in enumerate(metadata):
            self.metadata[self.index.ntotal - len(metadata) + i] = meta
        logging.info("✅ Added %d embeddings to the FAISS index.", len(embeddings))

    def normalize_vector(self, vector):
        """
        Normalize a vector to unit length using L2 normalization.
        Args:
            vector (list or np.array): The embedding vector to normalize.
        Returns:
            np.array: The normalized embedding vector.        
        """
        logging.debug("➡️ Normalizing vector...")
        vector = np.array(vector).reshape(1, -1).astype("float32")
        faiss.normalize_L2(vector)
        return vector

    def search(self, query_embedding, k=5):
        """
        Search the index for the k nearest neighbors of the query embedding.
        Args:
            query_embedding (list or np.array): The embedding vector to search for.
            k (int): The number of nearest neighbors to return.
        Returns:
            list of tuples: A list of (metadata, distance) tuples for the nearest neighbors.        
        """
        if not self.is_trained:
            raise RuntimeError("Index must be trained before searching.")
        logging.debug("🔍 Searching FAISS index for %d neighbors...", k)
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, k)
        logging.debug("✅ Found %d results.", len(indices[0]))
        return [
            (self.metadata[int(i)], distances[0][j]) for j, i in enumerate(indices[0])
        ]

    @classmethod
    def load_from_file(cls, file_path, verbose=False):
        """
        Load embeddings from a JSON file and create a FAISS indexer instance.
        Args:
            file_path (str): The path to the JSON file containing embeddings and metadata.
        Returns:
            FAISSIndexer: An instance of FAISSIndexer with the loaded embeddings and metadata.        
        """
        logging.info("📂 Loading embeddings from file: %s", file_path)
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not data:
            raise ValueError("No embeddings found in the dataset!")

        embedding_dim = len(data[0]["embedding"])
        indexer = cls(embedding_dim, verbose=verbose)

        embeddings = []
        metadata = []
        for item in data:
            embeddings.append(item["embedding"])
            metadata.append(
                {
                    "chunk_id": item["chunk_id"],
                    "filename": item["filename"],
                    "text": item["text"],
                    "tokens": item["tokens"],
                }
            )

        indexer.train(embeddings)
        indexer.add_embeddings(embeddings, metadata)
        logging.info("📚 Loaded %d embeddings from file.", len(embeddings))
        return indexer

    def save(self, file_path):
        """
        Save the FAISS index and its metadata to files.
        Args:
            file_path (str): The path to save the FAISS index file.
        Returns:     None
        """
        logging.info("💾 Saving FAISS index to: %s", file_path)
        faiss.write_index(self.index, file_path)
        with open(file_path + ".metadata", "w", encoding="utf-8") as f:
            json.dump(self.metadata, f)
        logging.info("✅ FAISS index saved successfully.")

    @classmethod
    def load(cls, file_path):
        """
        Load a FAISS index and its metadata from files.
        Args:
            file_path (str): The path to the FAISS index file.
        Returns:
            FAISSIndexer: An instance of FAISSIndexer with the loaded index and metadata.
        """
        logging.info("📂 Loading FAISS index from: %s", file_path)
        index = faiss.read_index(file_path)
        with open(file_path + ".metadata", "r", encoding="utf-8") as f:
            metadata = json.load(f)

        indexer = cls(index.d)
        indexer.index = index
        indexer.metadata = metadata
        indexer.is_trained = True
        logging.info("✅ FAISS index loaded successfully.")
        return indexer
