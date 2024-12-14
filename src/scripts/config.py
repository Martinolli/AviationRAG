from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
PROCESSED_DIR = BASE_DIR / "data" / "processed"
CHUNKED_DIR = PROCESSED_DIR / "chunked_documents"
TEXT_OUTPUT_DIR = PROCESSED_DIR / "ProcessedText"
TEXT_EXPANDED_DIR = PROCESSED_DIR / "ProcessedTextExpanded"
PKL_FILENAME = BASE_DIR / "data" / "raw" / "aviation_corpus.pkl"
EMBEDDINGS_FILE = BASE_DIR / "data" / "embeddings" / "aviation_embeddings.json"
