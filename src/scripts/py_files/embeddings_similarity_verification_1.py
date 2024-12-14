import json
import numpy as np
import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize the OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-ada-002"):
    """Generate embedding using OpenAI's updated client."""
    try:
        # Ensure the input is wrapped as a list
        response = client.embeddings.create(
            input=[text],  # OpenAI API requires input as a list
            model=model
        )
        # Extract the embedding
        return response.data[0].embedding
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

try:
    with open(embeddings_path, 'r') as file:
        data = json.load(file)
except FileNotFoundError:
    print(f"Embedding file not found at {embeddings_path}. Please check the path.")
    exit()

# Filter by filename and chunk length
filtered_data = [item for item in data if item['filename'] == '14cfr_safety_management_systems.pdf'
                 and 100 <= len(item['text'].split()) <= 400
                 ]


print(f"Number of filtered chunks: {len(filtered_data)}")  # Debug print

# Ask user for a query statement
query_text = input("Enter your query statement: ")
query_embedding = get_embedding(query_text)

# Validate the query embedding
if query_embedding is None:
    print("Failed to generate query embedding. Exiting.")
    exit()

# Compute similarity for each stored embedding
similarities = []
for item in filtered_data:
    print(f"Document embedding: {item['embedding']}")  # Debug print
    similarity = cosine_similarity(query_embedding, item["embedding"])
    print(f"Similarity: {similarity}")  # Debug print
    similarities.append({
        "chunk_id": item["chunk_id"],
        "filename": item["filename"],
        "text": item["text"],
        "similarity": similarity
    })

# Sort results by similarity
similarities = sorted(similarities, key=lambda x: x["similarity"], reverse=True)

# Display all filtered and sorted results with pagination
page_size = 3
for i in range(0, len(similarities), page_size):
    page = similarities[i:i+page_size]
    print("\nFiltered and Sorted Chunks by Similarity:")
    print(f"\nPage {i // page_size + 1}:\n")
    for result in page:
        print(f"Chunk ID: {result['chunk_id']}, Filename: {result['filename']}, Similarity: {result['similarity']:.4f}")
        print(f"Text: {result['text']}\n")

    if i + page_size < len(similarities):
        input("Press Enter to see the next page...")

