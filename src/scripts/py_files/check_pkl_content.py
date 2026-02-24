import pickle
import os

from config import PKL_FILENAME

def check_pkl_content():
    pkl_path = PKL_FILENAME
    
    if not pkl_path.exists():
        print(f"Error: The file {pkl_path} does not exist.")
        return
    
    try:
        with open(pkl_path, 'rb') as file:
            corpus = pickle.load(file)
        
        print(f"Successfully loaded aviation_corpus.pkl")
        print(f"Number of documents: {len(corpus)}")
        
        # Print details of the first few documents
        for i, doc in enumerate(corpus[:]):
            print(f"\nDocument {i + 1}:")
            print(f"Filename: {doc.get('filename', 'N/A')}")
            print(f"Text length: {len(doc.get('text', ''))}")
            print(f"Number of tokens: {len(doc.get('tokens', []))}")
            print(f"Number of entities: {len(doc.get('entities', []))}")
            print(f"Number Section references: {len(doc.get('section_references', []))}")
            print(f"Number of personal names: {len(doc.get('personal_names', []))}")
            print(f"Category: {doc.get('category', 'N/A')}")
            print(f"Number of Pos-Tags: {len(doc.get('pos_tags', []))}")
    
    except Exception as e:
        print(f"An error occurred while reading the file: {e}")

if __name__ == "__main__":
    check_pkl_content()
