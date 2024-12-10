import pickle
import json
import os

# Define absolute paths
base_dir = r'C:\Users\Aspire5 15 i7 4G2050\Project_2\AviationRAG'
pkl_path = os.path.join(base_dir, 'data', 'raw', 'aviation_corpus.pkl')
json_path = os.path.join(base_dir, 'data', 'processed', 'aviation_corpus.json')

# Load the pickle file
with open(pkl_path, 'rb') as file:
    corpus = pickle.load(file)

# Convert the pickle file to a JSON format
with open(json_path, 'w', encoding='utf-8') as json_file:
    json.dump(corpus, json_file, ensure_ascii=False, indent=4)

print(f"Data successfully extracted and saved to {json_path}")

