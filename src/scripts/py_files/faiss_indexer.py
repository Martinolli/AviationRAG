import faiss
import numpy as np
import json
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class FAISSIndexer:
    def __init__(self, embedding_dim, nlist=10, verbose=False):  # Reduced nlist for smaller datasets
        self.verbose = verbose
        if self.verbose:
            logging.info(f"🛠️ Initializing FAISS index with embedding dimension: {embedding_dim}, nlist: {nlist}")
        self.embedding_dim = embedding_dim
        self.nlist = nlist
        self.quantizer = faiss.IndexFlatL2(embedding_dim)
        self.index = faiss.IndexIVFFlat(self.quantizer, embedding_dim, nlist, faiss.METRIC_L2)
        self.metadata = {}
        self.is_trained = False

    def train(self, embeddings):
        """Train the index with a set of embeddings."""
        logging.info(f"🏋️‍♀️ Training FAISS index with {len(embeddings)} embeddings...")
        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        self.index.train(embeddings_array)
        self.is_trained = True
        logging.info(f"✅ FAISS index trained with {len(embeddings)} embeddings.")

    def add_embeddings(self, embeddings, metadata):
        if not self.is_trained:
            raise RuntimeError("Index must be trained before adding embeddings.")
        logging.info(f"➕ Adding {len(embeddings)} embeddings to the FAISS index...")
        embeddings_array = np.array(embeddings).astype('float32')
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)
        for i, meta in enumerate(metadata):
            self.metadata[self.index.ntotal - len(metadata) + i] = meta
        logging.info(f"✅ Added {len(embeddings)} embeddings to the FAISS index.")

    def normalize_vector(self, vector):
        logging.debug("➡️ Normalizing vector...")
        vector = np.array(vector).reshape(1, -1).astype('float32')
        faiss.normalize_L2(vector)
        return vector
    
    def search(self, query_embedding, k=5):
        if not self.is_trained:
            raise RuntimeError("Index must be trained before searching.")
        logging.debug(f"🔍 Searching FAISS index for {k} neighbors...")
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        faiss.normalize_L2(query_embedding)
        distances, indices = self.index.search(query_embedding, k)
        logging.debug(f"✅ Found {len(indices[0])} results.")
        return [(self.metadata[int(i)], distances[0][j]) for j, i in enumerate(indices[0])]

    @classmethod
    def load_from_file(cls, file_path, verbose=False):
        logging.info(f"📂 Loading embeddings from file: {file_path}")
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not data:
            raise ValueError("No embeddings found in the dataset!")

        embedding_dim = len(data[0]['embedding'])
        indexer = cls(embedding_dim, verbose=verbose)

        embeddings = []
        metadata = []
        for item in data:
            embeddings.append(item['embedding'])
            metadata.append({
                'chunk_id': item['chunk_id'],
                'filename': item['filename'],
                'text': item['text'],
                'tokens': item['tokens']
            })
        
        indexer.train(embeddings)
        indexer.add_embeddings(embeddings, metadata)
        logging.info(f"📚 Loaded {len(embeddings)} embeddings from file.")
        return indexer

    def save(self, file_path):
        logging.info(f"💾 Saving FAISS index to: {file_path}")
        faiss.write_index(self.index, file_path)
        with open(file_path + '.metadata', 'w') as f:
            json.dump(self.metadata, f)
        logging.info(f"✅ FAISS index saved successfully.")

    @classmethod
    def load(cls, file_path):
        logging.info(f"📂 Loading FAISS index from: {file_path}")
        index = faiss.read_index(file_path)
        with open(file_path + '.metadata', 'r') as f:
            metadata = json.load(f)
        
        indexer = cls(index.d)
        indexer.index = index
        indexer.metadata = metadata
        indexer.is_trained = True
        logging.info(f"✅ FAISS index loaded successfully.")
        return indexer
