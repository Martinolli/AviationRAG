import os
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, filename='read_documents_update.log', format='%(asctime)s - %(levelname)s - %(message)s')

# Define paths
BASE_DIR = r'C:\Users\Aspire5 15 i7 4G2050\ProjectRAG\AviationRAG'
TEXT_OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'ProcessedText')
TEXT_EXPANDED_DIR = os.path.join(BASE_DIR, 'data', 'processed', 'ProcessedTextExpanded')
PKL_FILENAME = os.path.join(BASE_DIR, 'data', 'raw', 'aviation_corpus.pkl')

# Ensure directories exist
def ensure_directory_exists(directory_path):
    """Ensure the directory exists, and log its creation."""
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logging.info(f"Created directory: {directory_path}")
    else:
        logging.info(f"Directory already exists: {directory_path}")

# Ensure necessary directories
ensure_directory_exists(TEXT_OUTPUT_DIR)
ensure_directory_exists(TEXT_EXPANDED_DIR)
ensure_directory_exists(os.path.dirname(PKL_FILENAME))

# Processing logic
def process_documents():
    logging.info("Starting document processing...")
    # Simulate processing (add your logic here)
    logging.info("Document processing completed.")

if __name__ == '__main__':
    process_documents()
