from pathlib import Path

# File location: <project_root>/src/scripts/py_files/config.py
PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = PROJECT_ROOT / "src"
SCRIPTS_DIR = SRC_DIR / "scripts"
PY_SCRIPTS_DIR = SCRIPTS_DIR / "py_files"
JS_SCRIPTS_DIR = SCRIPTS_DIR / "js_files"

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
LOG_DIR = PROJECT_ROOT / "logs"
ASSETS_DIR = PROJECT_ROOT / "assets"
PICTURES_DIR = ASSETS_DIR / "pictures"
CHAT_DIR = PROJECT_ROOT / "chat"
CHAT_ID_DIR = PROJECT_ROOT / "chat_id"

DOCUMENTS_DIR = DATA_DIR / "documents"
CHUNKED_DIR = PROCESSED_DIR / "chunked_documents"
TEXT_OUTPUT_DIR = PROCESSED_DIR / "ProcessedText"
TEXT_EXPANDED_DIR = PROCESSED_DIR / "ProcessedTextExpanded"
PKL_FILENAME = RAW_DIR / "aviation_corpus.pkl"
JSON_CORPUS_FILE = PROCESSED_DIR / "aviation_corpus.json"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "aviation_embeddings.json"
ABBREVIATIONS_CSV = PROJECT_ROOT / "abbreviations.csv"
