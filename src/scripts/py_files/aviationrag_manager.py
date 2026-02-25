"""
AviationRAG Manager Script
This script orchestrates the entire AviationRAG processing pipeline,
executing each step sequentially while providing robust logging,
error handling, and execution time tracking.
Features:
- Executes each step of the pipeline with retries on failure.

"""


import subprocess
import logging
import time
import sys
import argparse
from tqdm import tqdm
from dotenv import load_dotenv

from config import LOG_DIR, PROJECT_ROOT

# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Ensure the log directory exists
LOG_DIR.mkdir(parents=True, exist_ok=True)

log_file_path = LOG_DIR / "aviationrag.log"

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
            logging.info("Attempt %d/%d - Starting %s...", attempt, max_retries, script_name)
            result = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
            )
            execution_time = time.time() - start_time
            logging.info("%s completed successfully in %.2f seconds.", script_name, execution_time)
            logging.info(result.stdout)
            return True, execution_time  # Success
        except subprocess.CalledProcessError as e:
            logging.error("Error in %s (Attempt %d): %s", script_name, attempt, e)
            logging.error("STDERR: %s", e.stderr)
            if attempt < max_retries:
                logging.info("Retrying %s in 5 seconds...", script_name)
                time.sleep(5)  # Wait before retrying
            else:
                logging.error("%s failed after %d attempts. Moving to next step.", script_name, max_retries)
                return False, 0  # Final failure
                
    return False, 0  # This line is reached only if all retries failed

def main(args):
    logging.info("--- AviationRAG Processing Pipeline Started ---")
    python_exec = sys.executable

    steps = [
        ([python_exec, "src/scripts/py_files/read_documents.py"], "Read Documents"),
        ([python_exec, "src/scripts/py_files/aviation_chunk_saver.py"], "Chunk Documents"),
        ([python_exec, "src/scripts/py_files/extract_pkl_to_json.py"], "Extract PKL to JSON"),
        ([python_exec, "src/scripts/py_files/check_pkl_content.py"], "Check PKL Content"),
        (["node", "src/scripts/js_files/check_new_chunks.js"], "Generate New Embeddings"),
        ([python_exec, "src/scripts/py_files/check_embeddings.py"], "Check Embeddings"),
        (["node", "src/scripts/js_files/check_new_embeddings.js"], "Store New Embeddings in AstraDB"),
        (["node", "src/scripts/js_files/check_astradb_content.js"], "Check AstraDB Content"),
        (["node", "src/scripts/js_files/check_astradb_consistency.js"], "Check AstraDB Consistency"),
        ([python_exec, "src/scripts/py_files/visualizing_data.py"], "Update Visualizing Data")
    ]
    
    failed_steps = []
    successful_steps = []
    total_time = 0
    
    with tqdm(total=len(steps), desc="Pipeline Progress", unit="step") as pbar:
        for command, script_name in steps:
            if args.step and script_name != args.step:
                pbar.update(1)
                continue
            
            success, execution_time = run_script(command, script_name)
            total_time += execution_time
            
            if success:
                successful_steps.append((script_name, execution_time))
            else:
                failed_steps.append(script_name)
            
            pbar.update(1)

    logging.info("\n--- Pipeline Summary ---")
    logging.info("Total execution time: %.2f seconds", total_time)
    logging.info("Successful steps: %d", len(successful_steps))
    logging.info("Failed steps: %d", len(failed_steps))

    if failed_steps:
        logging.error("Pipeline completed with errors in the following steps:")
        for step in failed_steps:
            logging.error(step)
    else:
        logging.info("Pipeline completed successfully without errors!")

    logging.info("\nDetailed Step Execution Times:")
    for step, time in successful_steps:
        logging.info("%s: %.2f seconds", step, time)

    logging.info("--- AviationRAG Processing Pipeline Finished ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AviationRAG Processing Pipeline")
    parser.add_argument("--step", help="Run a specific step in the pipeline")
    parser.add_argument("--verbose", action="store_true", help="Increase output verbosity")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    main(args)
