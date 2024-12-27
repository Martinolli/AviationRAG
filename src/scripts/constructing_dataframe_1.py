import pandas as pd
import json
import os

# Load aviation corpus
with open("data/raw/aviation_corpus.pkl", "rb") as f:
    corpus_data = pd.read_pickle(f)

# Load chunks
chunk_dir = "data/processed/chunked_documents/"
combined_chunks = []

for filename in os.listdir(chunk_dir):
    if filename.endswith('.json'):
        with open(os.path.join(chunk_dir, filename), 'r') as f:
            chunk_data = json.load(f)
            combined_chunks.extend(chunk_data.get('chunks', []))

# Load embeddings
with open("data/embeddings/aviation_embeddings.json", "r") as f:
    embeddings_data = json.load(f)

# Helper function to find embedding by chunk_id
def find_embedding(chunk_id, embeddings_data):
    for item in embeddings_data:
        if item.get("chunk_id") == chunk_id:  # Check if the chunk_id matches
            return item.get("embedding")
    return None  # Return None if no match is found


# Combine all data into a single DataFrame
# Combine all data into a single DataFrame
combined_data = []

for chunk_entry in chunk_data["chunks"]:  # Access the "chunks" list
    chunk_id = f"{chunk_data['filename']}-{chunk_entry['tokens']}"  # Create a unique chunk ID if needed
    text = chunk_entry["text"]
    filename = chunk_data["filename"]
    tokens = chunk_entry["tokens"]
    metadata = chunk_data.get("metadata", {})
    category = chunk_data.get("category", "")
    # embedding = find_embedding(chunk_id, embeddings_data)  # Use the helper function to find the embedding

    combined_data.append({
        "filename": filename,
        "chunk_id": chunk_id,
        "text_chunk": text,
        "tokens": tokens,
        "metadata": metadata,
        "category": category,
        # "embedding": embedding,
    })

# Convert to DataFrame
df = pd.DataFrame(combined_data)
df.to_csv("data/processed/combined_data.csv", index=False)
df.to_pickle("data/processed/combined_data.pkl")

