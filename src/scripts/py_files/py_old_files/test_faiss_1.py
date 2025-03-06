import faiss
import numpy as np
import json
import logging

# ✅ Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# ✅ Path to your embeddings file
EMBEDDINGS_FILE = "data/embeddings/aviation_embeddings.json"

# ✅ Define FAISS Index
class FAISSIndex:
    def __init__(self, embedding_dim):
        self.index = faiss.IndexFlatL2(embedding_dim)  # L2 distance for similarity search
        self.embeddings = []  # Store original embeddings for reference

    def add_embeddings(self, embeddings):
        """Adds new embeddings to FAISS index."""
        embeddings = np.array(embeddings, dtype=np.float32)

        if embeddings.shape[1] != self.index.d:
            logging.error(f"❌ Dimension mismatch! FAISS expects {self.index.d}, but embeddings have {embeddings.shape[1]}")
            return

        self.index.add(embeddings)
        logging.info(f"✅ Added {len(embeddings)} embeddings. FAISS Index now contains {self.index.ntotal} embeddings.")

        self.embeddings.extend(embeddings)
        logging.info(f"✅ Added {len(embeddings)} embeddings. FAISS Index now contains {self.index.ntotal} embeddings.")

    def search(self, query_embedding, k=5):
        """Finds top k nearest embeddings."""
        query_embedding = np.array(query_embedding, dtype=np.float32).reshape(1, -1)
        distances, indices = self.index.search(query_embedding, k)
        return indices[0], distances[0]

# ✅ Load Embeddings
def load_embeddings(file_path):
    """Loads aviation embeddings from JSON file."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            # Extract embeddings and create a metadata mapping
            embeddings = []
            metadata = {}
            for i, item in enumerate(data):
                embeddings.append(item['embedding'])
                metadata[i] = {
                    'chunk_id': item['chunk_id'],
                    'filename': item['filename'],
                    'text': item['text'],
                    'tokens': item['tokens']
                }
            # Convert embeddings to a numpy array
            embeddings_array = np.array(embeddings).astype('float32')
            
            if len(embeddings) == 0:
                logging.error("❌ No embeddings found in the dataset!")
                return []

            logging.info(f"📚 Loaded {len(embeddings)} embeddings from file.")
            return embeddings

    except Exception as e:
        logging.error(f"❌ Error loading embeddings: {e}")
        return []

# ✅ Load and Add Embeddings to FAISS
# Load embeddings first to get the correct dimension
embeddings = load_embeddings(EMBEDDINGS_FILE)

if len(embeddings) == 0:
    logging.error("❌ No embeddings loaded. FAISS cannot be initialized.")
    exit()

embedding_dim = len(embeddings[0])  # Detect the real dimension
logging.info(f"📏 Detected embedding dimension: {embedding_dim}")

# Initialize FAISS with the correct dimension
faiss_index = FAISSIndex(embedding_dim)
faiss_index.add_embeddings(embeddings)  # Add embeddings

# ✅ Test FAISS with a Sample Query
sample_query_embedding = embeddings[0] # if len(embeddings) > 0 else None

if sample_query_embedding is not None:
    logging.info(f"🔍 Testing FAISS retrieval with a sample query...")
    indices, distances = faiss_index.search(sample_query_embedding, k=5)

    logging.info(f"✅ FAISS Search Results: Indices - {indices}, Distances - {distances}")
else:
    logging.error("❌ No valid sample embedding to test FAISS search.")
