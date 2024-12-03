from transformers import BertModel, BertTokenizer
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import json
import os

# Load BERT model and tokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

def get_embedding(text):
    inputs = tokenizer(text, return_tensors='pt')
    outputs = model(**inputs)
    return outputs.last_hidden_state.mean(dim=1).detach().numpy()

# Your new statement
new_statement = "system safety"

# Generate embedding for the new statement
new_embedding = get_embedding(new_statement)

# Load the JSON file
base_dir = r'C:\Users\Aspire5 15 i7 4G2050\Project\AviationRAG'
embeddings_path = os.path.join(base_dir, 'data', 'embeddings', 'aviation_embeddings.json')
with open(embeddings_path, 'r') as file:
    data = json.load(file)

print("Structure of loaded data:")
print(type(data))
if isinstance(data, dict):
    print("Keys in the dictionary:", data.keys())
elif isinstance(data, list):
    print("Number of items in the list:", len(data))
    if len(data) > 0:
        print("Type of the first item:", type(data[0]))
        if isinstance(data[0], dict):
            print("Keys in the first item:", data[0].keys())

# Extract embeddings from the list of dictionaries
embeddings = np.array([item['embedding'] for item in data if item['filename'] == '14cfr_safety_management_systems.pdf'])

# Calculate similarity
similarity_scores = cosine_similarity(embeddings, new_embedding)

# Find the most similar chunk
most_similar_idx = np.argmax(similarity_scores)
most_similar_chunk = data[most_similar_idx]

print(f"Most similar chunk: {most_similar_chunk['text']}")
print(f"Similarity score: {similarity_scores[most_similar_idx]}")
