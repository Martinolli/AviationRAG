import pickle
import json
import os

# Define paths
pkl_path = os.path.join('data', 'raw', 'aviation_corpus.pkl')
json_path = os.path.join('data', 'processed', 'aviation_corpus.json')

# Load the PKL file
with open(pkl_path, 'rb') as file:
    corpus = pickle.load(file)

# Convert the data to JSON format
with open(json_path, 'w', encoding='utf-8') as json_file:
    json.dump(corpus, json_file, ensure_ascii=False, indent=4)

print(f"Data successfully extracted and saved to {json_path}")
