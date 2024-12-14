import pickle
import os
from pathlib import Path

def check_pkl_content():
    # Get the project root directory
    project_root = Path(__file__).resolve().parent.parent.parent
    
    # Define the path to the aviation_corpus.pkl file
    pkl_path = project_root / 'data' / 'raw' / 'aviation_corpus.pkl'
    
    if not pkl_path.exists():
        print(f"Error: The file {pkl_path} does not exist.")
        return
    
    try:
        with open(pkl_path, 'rb') as file:
            corpus = pickle.load(file)
        
        print(f"Successfully loaded aviation_corpus.pkl")
        print(f"Number of documents: {len(corpus)}")
        
        # Print details of the first few documents
        for i, doc in enumerate(corpus[:5]):
            print(f"\nDocument {i + 1}:")
            print(f"Filename: {doc.get('filename', 'N/A')}")
            print(f"Text length: {len(doc.get('text', ''))}")
            print(f"Number of tokens: {len(doc.get('tokens', []))}")
            print(f"Number of entities: {len(doc.get('entities', []))}")
            print(f"Number of personal names: {len(doc.get('personal_names', []))}")
    
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    check_pkl_content()