# update_corpus.py
import os
import pickle
import logging
from .scraper_manager import ScraperManager

# Define paths
base_dir = r'C:\Users\Aspire5 15 i7 4G2050\ProjectRAG\AviationRAG'
corpus_path = os.path.join(base_dir, 'data', 'raw', 'aviation_corpus.pkl')
download_dir = os.path.join(base_dir, 'data', 'raw', 'scraped')

def update_corpus_with_scraped_documents(scrapers_to_run=None):
    """Update the aviation_corpus.pkl with scraped documents"""
    # Initialize scraper manager
    manager = ScraperManager(download_dir)
    
    # Run scrapers and get processed documents
    scraped_documents = []
    if scrapers_to_run:
        for scraper_name in scrapers_to_run:
            docs = manager.run_specific_scraper(scraper_name)
            scraped_documents.extend(docs)
    else:
        scraped_documents = manager.run_all_scrapers()
    
    # Format documents to match the existing corpus structure
    formatted_documents = []
    for doc in scraped_documents:
        formatted_doc = {
            'filename': doc['metadata'].get('filename', 'unknown'),
            'text': doc['text'],
            'metadata': doc['metadata'],
            'category': doc['metadata'].get('source', 'unknown')
            # Add other fields as needed to match your existing structure
        }
        formatted_documents.append(formatted_doc)
    
    # Load existing corpus
    existing_documents = []
    if os.path.exists(corpus_path):
        try:
            with open(corpus_path, 'rb') as f:
                existing_documents = pickle.load(f)
        except Exception as e:
            logging.error(f"Error loading corpus: {e}")
    
    # Combine and save updated corpus
    updated_corpus = existing_documents + formatted_documents
    
    try:
        with open(corpus_path, 'wb') as f:
            pickle.dump(updated_corpus, f)
        logging.info(f"Updated corpus with {len(formatted_documents)} scraped documents")
        return True
    except Exception as e:
        logging.error(f"Error saving corpus: {e}")
        return False

def main():
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Update corpus with all scrapers
    success = update_corpus_with_scraped_documents()
    
    if success:
        print("Successfully updated aviation corpus with scraped documents")
    else:
        print("Failed to update aviation corpus")

if __name__ == '__main__':
    main()