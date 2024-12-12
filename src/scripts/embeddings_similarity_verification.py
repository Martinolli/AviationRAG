import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up the OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-ada-002"):
    """
    Generate embedding using OpenAI's updated client.

    Args:
        text (str): The input text to generate an embedding for.
        model (str): The embedding model to use.

    Returns:
        list: The generated embedding.
    """
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

# Load embeddings from a JSON file
def load_embeddings(file_path):
    """
    Load embeddings from a JSON file.

    Args:
        file_path (str): Path to the JSON file containing embeddings.

    Returns:
        list: List of embeddings with metadata.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    return data

# Compute cosine similarity between a query and a list of embeddings
def compute_cosine_similarity(query_embedding, embeddings):
    """
    Compute cosine similarity between a query embedding and a list of embeddings.

    Args:
        query_embedding (list): The embedding for the query.
        embeddings (list): List of embeddings to compare against.

    Returns:
        list: List of cosine similarity scores.
    """
    query_vector = np.array(query_embedding).reshape(1, -1)
    embedding_vectors = np.array([item['embedding'] for item in embeddings])

    similarities = cosine_similarity(query_vector, embedding_vectors)[0]
    return similarities

# Filter and rank embeddings by similarity
def filter_and_rank_embeddings(embeddings, similarities, top_n=10, filename_filter=None):
    """
    Filter and rank embeddings based on similarity scores.

    Args:
        embeddings (list): List of embeddings with metadata.
        similarities (list): Corresponding similarity scores.
        top_n (int): Number of top results to return.
        filename_filter (str): Filter results by filename (optional).

    Returns:
        list: Top N ranked embeddings with metadata and similarity scores.
    """
    results = [
        {
            'chunk_id': emb['chunk_id'],
            'filename': emb['filename'],
            'text': emb['text'],
            'similarity': sim
        }
        for emb, sim in zip(embeddings, similarities)
        if filename_filter is None or filename_filter in emb['filename']
    ]

    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    return results[:top_n]

def generate_response(context, query):
    """
    Generate a response using OpenAI.

    Args:
        context (str): The context string generated from retrieved chunks.
        query (str): The user query.

    Returns:
        str: The generated response from OpenAI.
    """
    prompt = f"""
    Context:
    {context}

    Question:
    {query}

    Provide a detailed response based on the context above.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

# Main function to test similarity and generate responses
if __name__ == "__main__":
    # Path to the JSON file containing embeddings
    EMBEDDINGS_FILE = "data/embeddings/aviation_embeddings.json"

    # Query and parameters
    QUERY_TEXT = input("Enter your query text: ")
    TOP_N = 10
    FILENAME_FILTER = None  # Optional: filter results by filename

    try:
        # Generate embedding for the query
        print("Generating query embedding...")
        QUERY_EMBEDDING = get_embedding(QUERY_TEXT)
        if QUERY_EMBEDDING is None:
            raise ValueError("Failed to generate query embedding")

        # Load embeddings
        print("Loading embeddings...")
        embeddings = load_embeddings(EMBEDDINGS_FILE)

        # Compute similarities
        print("Computing similarities...")
        similarities = compute_cosine_similarity(QUERY_EMBEDDING, embeddings)

        # Filter and rank results
        print("Filtering and ranking results...")
        top_results = filter_and_rank_embeddings(
            embeddings, similarities, top_n=TOP_N, filename_filter=FILENAME_FILTER
        )

        # Combine context from top results
        context = "\n".join([result['text'] for result in top_results])

        # Generate response from OpenAI
        print("Generating response...")
        response = generate_response(context, QUERY_TEXT)

        # Display response
        print("\nGenerated Response:")
        print(response)

    except Exception as e:
        print(f"Error: {e}")
