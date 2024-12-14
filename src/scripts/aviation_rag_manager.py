import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import json
import subprocess
from dotenv import load_dotenv
import pickle
import sys
from logging.handlers import RotatingFileHandler
# Import project-specific modules
from read_documents import read_documents_from_directory
from aviation_chunk_saver import save_documents_as_chunks
# Load environment variables
load_dotenv()

# Get the current script's directory
current_dir = Path(__file__).resolve().parent

# Navigate up to the project root (assuming the script is in src/scripts)
project_root = current_dir.parent.parent

# Set the DOCUMENTS_DIR to the existing data/documents folder
DOCUMENTS_DIR = project_root / 'data' / 'documents'

# Verify that the directory exists
if not DOCUMENTS_DIR.exists():
    raise FileNotFoundError(f"The documents directory does not exist: {DOCUMENTS_DIR}")

# Use this DOCUMENTS_DIR in your script
print(f"DOCUMENTS_DIR is set to: {DOCUMENTS_DIR}")


# Define directories and files
BASE_DIR = project_root
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKED_DIR = PROCESSED_DIR / "chunked_documents"
EMBEDDINGS_FILE = BASE_DIR / "data" / "embeddings" / "aviation_embeddings.json"
PROCESSED_FILES_PATH = BASE_DIR / "processed_files.json"
PROCESSED_TEXT_DIR = BASE_DIR / "data" / "processed" / "ProcessedText"
PROCESSED_TEXT_EXPANDED_DIR = BASE_DIR / "data" / "processed" / "ProcessedTextExpanded"

text_output_dir=PROCESSED_TEXT_DIR,
text_expanded_dir=PROCESSED_TEXT_EXPANDED_DIR,

# Configure logging
log_dir = project_root / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

log_file = log_dir / "aviation_rag_manager.log"
log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
log_handler = RotatingFileHandler(log_file, maxBytes=1024 * 1024, backupCount=5)
log_handler.setFormatter(log_formatter)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Use DEBUG to capture all messages

# Clear existing handlers to avoid duplication
logger.handlers.clear()
logger.addHandler(log_handler)

# Add console output for immediate feedback
logger.addHandler(logging.StreamHandler(sys.stdout))

# Test log
logger.info("Logging setup complete. This should appear in the file and console.")

def update_aviation_corpus():
    """
    Update the aviation_corpus.pkl file with newly processed documents.
    """
    logger.info("Updating aviation_corpus.pkl with new documents.")
    corpus_path = BASE_DIR / "data" / "raw" / "aviation_corpus.pkl"
    processed_dir = BASE_DIR / "data" / "processed" / "ProcessedText"
    expanded_dir = BASE_DIR / "data" / "processed" / "ProcessedTextExpanded"
    logger.info(f"Processed text output directory: {processed_dir}")
    logger.info(f"Processed text expanded directory: {expanded_dir}")

    # Load the existing corpus if available
    if corpus_path.exists():
        with open(corpus_path, 'rb') as file:
            existing_corpus = pickle.load(file)
        logger.info(f"Loaded existing corpus with {len(existing_corpus)} documents.")
    else:
        existing_corpus = []
    logger.info("No existing corpus found. Creating a new one.")

    # Update the corpus with new documents
    updated_corpus = read_documents_from_directory(
        directory_path=DOCUMENTS_DIR,
        text_output_dir=PROCESSED_TEXT_DIR,
        text_expanded_dir=PROCESSED_TEXT_EXPANDED_DIR,
        existing_documents=existing_corpus
    )

    # Save the updated corpus back to the PKL file
    with open(corpus_path, 'wb') as file:
        pickle.dump(updated_corpus, file)
    
    logger.info(f"aviation_corpus.pkl updated with {len(updated_corpus)} total documents.")


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
    logger.info(f"DOCUMENTS_DIR is set to: {DOCUMENTS_DIR}")
    logger.info(f"Does DOCUMENTS_DIR exist? {os.path.exists(DOCUMENTS_DIR)}")
    
    if os.path.exists(DOCUMENTS_DIR):
        all_files = os.listdir(DOCUMENTS_DIR)
        logger.info(f"Files in directory: {all_files}")
        
        for file in all_files:
            full_path = os.path.join(DOCUMENTS_DIR, file)
        logger.info(f"File: {file}, Is file: {os.path.isfile(full_path)}, Size: {os.path.getsize(full_path)} bytes")
    else:
        logger.error(f"The DOCUMENTS_DIR does not exist: {DOCUMENTS_DIR}")

    # Load processed files
    processed_files = load_processed_files(PROCESSED_FILES_PATH)
    logger.info(f"Loaded {len(processed_files)} processed files.")
    
    # Step 1: Process new documents
    logger.info("Processing new documents...")
    logger.info(f"Scanning directory: {DOCUMENTS_DIR}")

    # Log all files in the directory
    all_files = os.listdir(DOCUMENTS_DIR)
    logger.info(f"Files in directory: {all_files}")


    all_documents = read_documents_from_directory(
    directory_path=DOCUMENTS_DIR,
    text_output_dir=PROCESSED_DIR / "ProcessedText",
    text_expanded_dir=PROCESSED_DIR / "ProcessedTextExpanded"
)
    logger.info(f"Total documents found: {len(all_documents)}")

    # Filter out already processed documents
    new_documents = [doc for doc in all_documents if doc['filename'] not in processed_files]
    logger.info(f"New documents to process: {[doc['filename'] for doc in new_documents]}")

    if not new_documents:
        logger.info("No new documents to process. Exiting routine.")
        return
    else:
        logger.info(f"Found {len(new_documents)} new documents to process.")

    # Step 2: Create chunks
    logger.info("Creating text chunks...")
    save_documents_as_chunks(
        documents=new_documents,
        output_dir=CHUNKED_DIR,
        max_tokens=500,
        overlap=50
    )
    logger.info("Chunks created and saved.")

    # Step 3: Generate embeddings
    logger.info("Generating embeddings...")
    generate_embeddings(
        chunked_docs_path=CHUNKED_DIR,
        output_path=EMBEDDINGS_FILE
    )
    logger.info(f"Embeddings saved to {EMBEDDINGS_FILE}.")

    # Step 4: Store embeddings in database
    logger.info("Storing embeddings in Astra DB...")
    insert_embeddings_into_db(embeddings_path=EMBEDDINGS_FILE)
    logger.info("Embeddings successfully stored in Astra DB.")

    # Update processed files list
    processed_files.update([doc['filename'] for doc in new_documents])
    save_processed_files(PROCESSED_FILES_PATH, processed_files)
    logger.info(f"Updated processed files list. Total processed files: {len(processed_files)}.")

    # Update aviation_corpus.pkl after processing
    update_aviation_corpus()

    logger.info("Aviation RAG Manager completed.")

# Entry point
if __name__ == "__main__":
    try:
        aviation_rag_manager()
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
