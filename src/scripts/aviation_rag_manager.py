import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import json
import subprocess
from dotenv import load_dotenv

# Import project-specific modules
from read_documents import read_documents_from_directory
from aviation_chunk_saver import save_documents_as_chunks

# Load environment variables
load_dotenv()

# Configure logging
log_file = "aviation_rag_manager.log"
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_handler = RotatingFileHandler(log_file, maxBytes=1024*1024, backupCount=5)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(log_handler)

# Define directories and files
BASE_DIR = Path(__file__).resolve().parent.parent
DOCUMENTS_DIR = BASE_DIR / "data/documents"
PROCESSED_DIR = BASE_DIR / "data/processed"
CHUNKED_DIR = PROCESSED_DIR / "chunked_documents"
EMBEDDINGS_FILE = BASE_DIR / "data/embeddings/aviation_embeddings.json"
PROCESSED_FILES_PATH = BASE_DIR / "processed_files.json"

# Ensure necessary directories exist
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHUNKED_DIR.mkdir(parents=True, exist_ok=True)

# Function to run JavaScript files
def run_js_script(script_name, *args):
    script_path = BASE_DIR / "src" / "scripts" / script_name
    result = subprocess.run(["node", str(script_path)] + list(args), capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"Error running {script_name}: {result.stderr}")
        raise RuntimeError(f"Error running {script_name}")
    return result.stdout

# Modified functions to call JavaScript files
def generate_embeddings(chunked_docs_path, output_path):
    logger.info("Generating embeddings using Node.js script...")
    run_js_script("generate_embeddings.js", str(chunked_docs_path), str(output_path))
    logger.info(f"Embeddings saved to {output_path}.")

def insert_embeddings_into_db(embeddings_path):
    logger.info("Storing embeddings in Astra DB using Node.js script...")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            run_js_script("store_embeddings_astra.js", str(embeddings_path))
            logger.info("Embeddings successfully stored in Astra DB.")
            return
        except RuntimeError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Attempt {attempt + 1} failed. Retrying...")
            else:
                logger.error("Failed to store embeddings after multiple attempts.")
                raise

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
    logger.info("Starting Aviation RAG Manager.")

    # Load processed files
    processed_files = load_processed_files(PROCESSED_FILES_PATH)

    # Step 1: Process new documents
    logger.info("Processing new documents...")
    all_documents = read_documents_from_directory(
    directory_path=DOCUMENTS_DIR,
    text_output_dir=PROCESSED_DIR / "ProcessedTex",
    text_expanded_dir=PROCESSED_DIR / "ProcessedTextExpanded"
)

    # Filter out already processed documents
    new_documents = [doc for doc in all_documents if doc['filename'] not in processed_files]

    if not new_documents:
        logger.info("No new documents to process. Exiting routine.")
        return
    else:
        logger.info(f"Found {len(new_documents)} new documents to process.")

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
        logger.error(f"An error occurred: {e}", exc_info=True)
