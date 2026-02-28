"""Module for extracting pickle files to JSON format."""
import pickle
import json

from config import JSON_CORPUS_FILE, PKL_FILENAME

pkl_path = PKL_FILENAME
json_path = JSON_CORPUS_FILE

def extract_pkl_to_json(pkl_path, json_path):
    """
    Extract data from a pickle file and save it as a JSON file.
    
    Args:
        pkl_path: Path to the input pickle file.
        json_path: Path to the output JSON file.
    """
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pkl_path, 'rb') as file:
        corpus = pickle.load(file)
    with open(json_path, 'w', encoding='utf-8') as json_file:
        json.dump(corpus, json_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    extract_pkl_to_json(pkl_path, json_path)
    print(f"Data successfully extracted and saved to {json_path}")
