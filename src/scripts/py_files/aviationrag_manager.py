import subprocess
import logging
import os
import time
from dotenv import load_dotenv
import sys

# Load environment variables
load_dotenv()

# Define base directory
BASE_DIR = r'C:\Users\Aspire5 15 i7 4G2050\ProjectRAG\AviationRAG'

LOG_DIR = os.path.join(BASE_DIR, 'logs')  # Define the path to the logs folder

# Ensure the log directory exists
if not os.path.exists(LOG_DIR):
    os.makedirs(LOG_DIR)

log_file_path = os.path.join(LOG_DIR, 'aviationrag.log')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path, mode='w', encoding='utf-8'),  # Use UTF-8 encoding
        logging.StreamHandler(sys.stdout)  # Print logs to console
    ]
)

def run_script(command, script_name, max_retries=3):
    """Execute a script with logging, retries, and execution time tracking."""
    attempt = 0
    while attempt < max_retries:
        try:
            attempt += 1
            start_time = time.time()
            logging.info(f"🟡 Attempt {attempt}/{max_retries} - Starting {script_name}...")
            
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            
            execution_time = time.time() - start_time
            logging.info(f"✅ {script_name} completed successfully in {execution_time:.2f} seconds.")
            logging.info(result.stdout)
            return True  # Success
            
        except subprocess.CalledProcessError as e:
            logging.error(f"❌ Error in {script_name} (Attempt {attempt}): {e}")
            logging.error(f"🔍 STDERR: {e.stderr}")
            
            if attempt < max_retries:
                logging.info(f"🔄 Retrying {script_name} in 5 seconds...")
                time.sleep(5)  # Wait before retrying
            else:
                logging.error(f"⛔ {script_name} failed after {max_retries} attempts. Moving to next step.")
                return False  # Final failure
                
    return False  # This line is reached only if all retries failed

if __name__ == "__main__":
    logging.info("🚀 --- AviationRAG Processing Pipeline Started --- 🚀")
    
    steps = [
        ("python src/scripts/py_files/read_documents.py", "Read Documents"),
        ("python src/scripts/py_files/aviation_chunk_saver.py", "Chunk Documents"),
        ("python src/scripts/py_files/extract_pkl_to_json.py", "Extract PKL to JSON"),
        ("node src/scripts/js_files/generate_embeddings.js", "Generate Embeddings"),
        ("node src/scripts/js_files/store_embeddings_astra.js", "Store Embeddings in AstraDB"),
        ("node src/scripts/js_files/check_astradb_consistency.js", "Check AstraDB Consistency"),
        ("node src/scripts/js_files/check_astradb_content.js", "Check AstraDB Content"),
        ("python src/scripts/py_files/visualizing_data.py", "Update Visualizing Data")
    ]
    
    failed_steps = []
    
    for command, script_name in steps:
        if not run_script(command, script_name):
            failed_steps.append(script_name)

    if failed_steps:
        logging.error("❗ Pipeline completed with errors in the following steps:")
        for step in failed_steps:
            logging.error(f"⛔ {step}")
    else:
        logging.info("✅ Pipeline completed successfully without errors!")

    logging.info("🏁 --- AviationRAG Processing Pipeline Finished --- 🏁")
