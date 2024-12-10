import json
import numpy as np
import os
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up the OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-ada-002"):
    """Generate embedding using OpenAI's updated client."""
    try:
        response = client.embeddings.create(
            input=[text],  # OpenAI API requires input as a single string or list
            model=model
        )
        # Extract the embedding
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def load_embeddings(file_path):
    """Load embeddings from a JSON file."""
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

def filter_embeddings(embeddings, filename_filter=None, chunk_size_range=(100, 500)):
    """Filter embeddings by filename and chunk size."""
    filtered = []
    for embedding in embeddings:
        chunk_size = len(embedding['text'].split())
        if filename_filter and embedding['filename'] != filename_filter:
            continue
        if chunk_size < chunk_size_range[0] or chunk_size > chunk_size_range[1]:
            continue
        filtered.append(embedding)
    return filtered

def cosine_similarity(vec1, vec2):
    """Compute the cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    return dot_product / (norm_vec1 * norm_vec2)

def compute_similarity(query_embedding, document_embeddings):
    """Compute cosine similarity between query embedding and document embeddings."""
    similarities = []
    for embedding in document_embeddings:
        similarity = cosine_similarity(np.array(query_embedding), np.array(embedding['embedding']))
        similarities.append((embedding, similarity))
    return sorted(similarities, key=lambda x: x[1], reverse=True)

def paginate_results(results, page_size=5):
    """Paginate results for easier viewing."""
    for i in range(0, len(results), page_size):
        yield results[i:i+page_size]

if __name__ == "__main__":
    # Paths and inputs
    EMBEDDINGS_FILE = "data/embeddings/aviation_embeddings.json"
    QUERY_TEXT = input("Enter your query text: ")
    FILENAME_FILTER = input("Enter filename to filter (or leave blank): ")

    # Load embeddings
    embeddings = load_embeddings(EMBEDDINGS_FILE)

    # Filter embeddings by filename and chunk size
    filtered_embeddings = filter_embeddings(embeddings, filename_filter=FILENAME_FILTER)

    # Generate query embedding
    query_embedding = get_embedding(QUERY_TEXT)

    # Compute similarity
    results = compute_similarity(query_embedding, filtered_embeddings)

    # Display results
    print(f"Top results for query: {QUERY_TEXT}")
    for page in paginate_results(results):
        for result, similarity in page:
            print(f"Filename: {result['filename']}, Similarity: {similarity:.4f}")
            print(f"Chunk: {result['text']}\n")
        input("Press Enter to view the next page...")
