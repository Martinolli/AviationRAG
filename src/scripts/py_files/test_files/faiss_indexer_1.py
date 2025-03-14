import faiss
import numpy as np
import json
import logging

class FAISSIndexer:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatL2(embedding_dim)
        self.metadata = {}

    def add_embeddings(self, embeddings, metadata):
        embeddings_array = np.array(embeddings).astype('float32')
        # ✅ Normalize embeddings before adding to FAISS
        faiss.normalize_L2(embeddings_array)
        self.index.add(embeddings_array)
        for i, meta in enumerate(metadata):
            self.metadata[i] = meta  # ✅ Use explicit indexing to prevent misalignment

    def normalize_vector(self, vector):
        """
        Normalize the input vector using L2 normalization.
        """
        vector = np.array(vector).reshape(1, -1).astype('float32')
        faiss.normalize_L2(vector)
        return vector
    
    def search(self, query_embedding, k=5):
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        return [(self.metadata[int(i)], distances[0][j]) for j, i in enumerate(indices[0])]

    @classmethod
    def load_from_file(cls, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        if not data:
            raise ValueError("No embeddings found in the dataset!")

        embedding_dim = len(data[0]['embedding'])
        indexer = cls(embedding_dim)

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

        indexer.add_embeddings(embeddings, metadata)
        logging.info(f"📚 Loaded {len(embeddings)} embeddings from file.")
        return indexer

    def save(self, file_path):
        faiss.write_index(self.index, file_path)
        with open(file_path + '.metadata', 'w') as f:
            json.dump(self.metadata, f)

    @classmethod
    def load(cls, file_path):
        index = faiss.read_index(file_path)
        with open(file_path + '.metadata', 'r') as f:
            metadata = json.load(f)
        
        indexer = cls(index.d)
        indexer.index = index
        indexer.metadata = metadata
        return indexer