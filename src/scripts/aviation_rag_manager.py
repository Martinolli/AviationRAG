import os
import logging
from pathlib import Path
import json
from dotenv import load_dotenv

# Import project-specific modules
from read_documents import read_documents_from_directory
from aviation_chunk_saver import save_documents_as_chunks
from generate_embeddings import generate_embeddings
from store_embeddings_astra import insert_embeddings_into_db

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, filename="aviation_rag_manager.log", format="%(asctime)s - %(levelname)s - %(message)s")

# Define directories and files
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "data/documents"
PROCESSED_DIR = BASE_DIR / "data/processed"
CHUNKED_DIR = PROCESSED_DIR / "chunked_documents"
EMBEDDINGS_FILE = BASE_DIR / "data/embeddings/aviation_embeddings.json"
PROCESSED_FILES_PATH = BASE_DIR / "processed_files.json"

# Ensure necessary directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHUNKED_DIR.mkdir(parents=True, exist_ok=True)

# Utility functions for tracking processed files
def load_processed_files(file_path):
    """Load processed filenames from a JSON file."""
    if not os.path.exists(file_path):
        return set()
    with open(file_path, 'r') as file:
        return set(json.load(file))

def save_processed_files(file_path, filenames):
    """Save processed filenames to a JSON file."""
    with open(file_path, 'w') as file:
        json.dump(list(filenames), file)

# Main routine
def aviation_rag_manager():
    logging.info("Starting Aviation RAG Manager.")

    # Load processed files
    processed_files = load_processed_files(PROCESSED_FILES_PATH)

    # Step 1: Process new documents
    logging.info("Processing new documents...")
    new_documents = read_documents_from_directory(
        directory_path=DOCUMENTS_DIR,
        processed_filenames=processed_files,
        text_output_dir=PROCESSED_DIR / "ProcessedTex",
        text_expanded_dir=PROCESSED_DIR / "ProcessedTextExpanded"
    )

    if not new_documents:
        logging.info("No new documents to process. Exiting routine.")
        return

    # Step 2: Create chunks
    logging.info("Creating text chunks...")
    save_documents_as_chunks(
        documents=new_documents,
        output_dir=CHUNKED_DIR,
        max_tokens=500,
        overlap=50
    )
    logging.info("Chunks created and saved.")

    # Step 3: Generate embeddings
    logging.info("Generating embeddings...")
    generate_embeddings(
        chunked_docs_path=CHUNKED_DIR,
        output_path=EMBEDDINGS_FILE
    )
    logging.info(f"Embeddings saved to {EMBEDDINGS_FILE}.")

    # Step 4: Store embeddings in database
    logging.info("Storing embeddings in Astra DB...")
    insert_embeddings_into_db(embeddings_path=EMBEDDINGS_FILE)
    logging.info("Embeddings successfully stored in Astra DB.")

    # Update processed files list
    processed_files.update([doc['filename'] for doc in new_documents])
    save_processed_files(PROCESSED_FILES_PATH, processed_files)
    logging.info(f"Updated processed files list. Total processed files: {len(processed_files)}.")

    logging.info("Aviation RAG Manager completed.")

# Entry point
if __name__ == "__main__":
    try:
        aviation_rag_manager()
    except Exception as e:
        logging.error(f"An error occurred: {e}", exc_info=True)
