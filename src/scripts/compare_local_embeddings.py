import json
import numpy as np
import os
import openai
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Retrieve the API key from the environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

print(f"Loaded API Key: {openai.api_key}")

import openai

def get_embedding(text, model="text-embedding-ada-002"):
    """Generate embedding for the given text using OpenAI's updated API."""
    try:
        # Ensure the input is wrapped in a list (as required by the API)
        response = openai.Embedding.create(
            input=text if isinstance(text, list) else [text],
            model=model
        )
        # Return the embedding vector from the response
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None


def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0  # Handle zero-norm vectors
    return np.dot(vec1, vec2) / (norm1 * norm2)

# Load stored embeddings from your JSON file
base_dir = r'C:\Users\Aspire5 15 i7 4G2050\Project\AviationRAG'
embeddings_path = os.path.join(base_dir, 'data', 'embeddings', 'aviation_embeddings.json')
with open(embeddings_path, 'r') as file:
    data = json.load(file)

# Filter the stored data (if needed)
filtered_data = [item for item in data]  # Adjust filters if necessary

# Ask user for a query statement
query_text = input("Enter your query statement: ")
query_embedding = get_embedding(query_text)

# Validate the embedding
if query_embedding is None:
    print("Failed to generate query embedding. Exiting.")
    exit()

# Compute similarity for each stored embedding
similarities = []
for item in filtered_data:
    similarity = cosine_similarity(query_embedding, item["embedding"])
    similarities.append({
        "chunk_id": item["chunk_id"],
        "filename": item["filename"],
        "text": item["text"],
        "similarity": similarity
    })

# Sort results by similarity
similarities = sorted(similarities, key=lambda x: x["similarity"], reverse=True)

# Display the top 5 matches
print("\nTop 5 similar chunks:")
for result in similarities[:5]:
    print(f"Chunk ID: {result['chunk_id']}, Filename: {result['filename']}, Similarity: {result['similarity']:.4f}")
    print(f"Text: {result['text']}\n")
