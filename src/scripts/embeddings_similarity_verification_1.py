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
            input=[text],
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def load_embeddings(file_path, batch_size=1000):
    """
    Load embeddings from a JSON file in batches to improve performance.

    Args:
        file_path (str): Path to the JSON file containing embeddings.
        batch_size (int): Number of embeddings to load at a time.

    Returns:
        generator: Generator yielding embeddings in batches.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

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

    return cosine_similarity(query_vector, embedding_vectors)[0]

def filter_and_rank_embeddings(embeddings, similarities, top_n=10):
    """
    Filter and rank embeddings based on similarity scores.

    Args:
        embeddings (list): List of embeddings with metadata.
        similarities (list): Corresponding similarity scores.
        top_n (int): Number of top results to return.

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
    ]

    # Sort results by similarity
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

if __name__ == "__main__":
    EMBEDDINGS_FILE = "data/embeddings/aviation_embeddings.json"

    QUERY_TEXT = input("Enter your query text: ")
    TOP_N = 10

    try:
        print("Generating query embedding...")
        query_embedding = get_embedding(QUERY_TEXT)
        if query_embedding is None:
            raise ValueError("Failed to generate query embedding")

        print("Loading embeddings...")
        top_results = []
        for batch in load_embeddings(EMBEDDINGS_FILE):
            # Compute similarities for this batch
            print(f"Processing batch of {len(batch)} embeddings...")
            similarities = compute_cosine_similarity(query_embedding, batch)

            # Filter and rank top results
            top_results.extend(filter_and_rank_embeddings(batch, similarities, top_n=TOP_N))

        # Combine context from top N results
        unique_texts = set()
        combined_context = ""
        for result in sorted(top_results, key=lambda x: x['similarity'], reverse=True)[:TOP_N]:
            if result['text'] not in unique_texts:  # Prevent duplicate context
                unique_texts.add(result['text'])
                combined_context += f"{result['text']}\n"

        print("Generating response...")
        response = generate_response(combined_context, QUERY_TEXT)

        print("\nGenerated Response:")
        print(response)

    except Exception as e:
        print(f"Error: {e}")
