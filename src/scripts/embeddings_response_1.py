import json
import logging
import time
import random
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up OpenAI API client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def expand_query(query):
    """Expand the user query with relevant aviation-specific terms."""
    prompt = f"Expand the following aviation-related query with technical terms: {query}"
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error expanding query: {e}")
        return query

def get_embedding(text):
    """Generate an embedding for the given text."""
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return None

def load_embeddings(file_path):
    """Load embeddings from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, list) and all(isinstance(item, dict) for item in data):
            return data
        raise ValueError("Loaded data is not in expected format (list of dictionaries)")
    except (FileNotFoundError, json.JSONDecodeError, ValueError) as e:
        logging.error(f"Error loading embeddings: {e}")
        return []

def compute_cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    try:
        vec1, vec2 = np.array(vec1), np.array(vec2)
        if vec1.shape != vec2.shape:
            raise ValueError("Vectors must have the same dimension")
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    except Exception as e:
        logging.error(f"Error computing cosine similarity: {e}")
        return 0.0

def filter_and_rank_embeddings(embeddings, similarities, top_n=10, min_similarity=0.5):
    """Filter and rank embeddings based on similarity scores."""
    avg_similarity = np.mean(similarities)
    threshold = max(avg_similarity, min_similarity)
    return sorted(
        [
            {**emb, 'similarity': sim}
            for emb, sim in zip(embeddings, similarities)
            if sim > threshold and isinstance(emb, dict) and 'embedding' in emb
        ],
        key=lambda x: x['similarity'],
        reverse=True
    )[:top_n]

def generate_response(context, query, model):
    """Generate a response using OpenAI."""
    prompt = f"""
    You are an AI specialized in aviation. Answer the user's question based on the provided context:
    Context:
    {context}

    Question:
    {query}
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=2000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Error generating response: {e}")
        return "I encountered an error generating a response. Please try again."

def chat_loop():
    EMBEDDINGS_FILE = "data/embeddings/aviation_embeddings.json"
    MODEL = "gpt-3.5-turbo"
    chat_history = []
    max_history = 5

    print("Welcome to AviationAI! Type 'exit' to quit.")

    embeddings = load_embeddings(EMBEDDINGS_FILE)
    if not embeddings:
        return

    while True:
        query = input("\nUser: ")
        if query.lower() == 'exit':
            print("Goodbye!")
            break

        expanded_query = expand_query(query)
        query_embedding = get_embedding(expanded_query)
        if query_embedding is None:
            continue

        similarities = [compute_cosine_similarity(query_embedding, emb['embedding']) for emb in embeddings if 'embedding' in emb]
        top_results = filter_and_rank_embeddings(embeddings, similarities)
        context = "\n".join([result['text'] for result in top_results])

        response = generate_response(context, query, MODEL)
        print("\nAviationAI:", response)

        chat_history.append((query, response))
        if len(chat_history) > max_history:
            chat_history = chat_history[-max_history:]

if __name__ == "__main__":
    chat_loop()
