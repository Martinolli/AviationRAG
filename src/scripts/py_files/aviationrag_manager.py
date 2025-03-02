import subprocess
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define base directory
BASE_DIR = r'C:\Users\Aspire5 15 i7 4G2050\ProjectRAG\AviationRAG'

LOG_DIR = os.path.join(BASE_DIR, 'logs')  # Define the path to the logs folder

log_file_path = os.path.join(LOG_DIR, 'aviationrag.log')

# Ensure the log directory exists
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w'),
        logging.StreamHandler()
    ]
)

def run_script(command, script_name):
    """Execute a script and handle errors."""
    try:
        logging.info(f"Starting {script_name}...")
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        logging.info(f"{script_name} completed successfully.")
        logging.info(result.stdout)
    except subprocess.CalledProcessError as e:
        logging.error(f"Error running {script_name}: {e}")
        logging.error(e.stderr)
        print(f"❌ {script_name} failed. Check aviationrag.log for details.")
        return False
    return True

if __name__ == "__main__":
    logging.info("--- AviationRAG Processing Pipeline Started ---")
    
    # Step 1: Read documents and generate aviation_corpus.pkl
    if not run_script("python src/scripts/py_files/read_documents.py", "Read Documents"):
        exit(1)
    
    # Step 2: Chunk documents
    if not run_script("python src/scripts/py_files/aviation_chunk_saver.py", "Chunk Documents"):
        exit(1)
    
    # Step 3: Convert PKL to JSON
    if not run_script("python src/scripts/py_files/extract_pkl_to_json.py", "Extract PKL to JSON"):
        exit(1)
    
    # Step 4: Generate embeddings
    if not run_script("node src/scripts/js_files/generate_embeddings.js", "Generate Embeddings"):
        exit(1)
    
    # Step 5: Store embeddings in AstraDB
    if not run_script("node src/scripts/js_files/store_embeddings_astra.js", "Store Embeddings in AstraDB"):
        exit(1)
    
    # Step 6: Validate database consistency
    if not run_script("node src/scripts/js_files/check_astradb_consistency.js", "Check AstraDB Consistency"):
        exit(1)

    # Step 7: Check AstraDB Content
    if not run_script("node src/scripts/js_files/check_astradb_content.js", "Check AstraDB Content"):
        exit(1)

    # Step 8: Update the visualizing data
    if not run_script("python src/scripts/py_files/visualizing_data.py", "Update Visualizing Data"):
        exit(1)
    
    logging.info("--- AviationRAG Processing Pipeline Completed Successfully ---")
    print("✅ AviationRAG processing pipeline completed!")
