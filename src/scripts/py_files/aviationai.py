import logging
import os
import numpy as np
import json
import time
import uuid
import sys
import re
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from faiss_indexer import FAISSIndexer
import tiktoken
import pdfplumber
from docx import Document as DocxDocument

from chat_db import retrieve_chat_from_db, store_chat_in_db
from config import CHAT_DIR, CHAT_ID_DIR, DOCUMENTS_DIR, EMBEDDINGS_FILE as EMBEDDINGS_PATH, LOG_DIR, PROJECT_ROOT

# ✅ Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# ✅ Define base paths
log_dir = LOG_DIR
chat_dir = CHAT_DIR
chat_id = CHAT_ID_DIR

# ✅ Ensure directories exist
for directory in [log_dir, chat_dir, chat_id]:
    directory.mkdir(parents=True, exist_ok=True)

# ✅ Configure logging
info_log_path = log_dir / "info.log"
error_log_path = log_dir / "error.log"
performance_log_path = log_dir / "performance.log"

log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# ✅ Setup log handlers
info_log = logging.FileHandler(info_log_path, encoding="utf-8")
error_log = logging.FileHandler(error_log_path, encoding="utf-8")
performance_log = logging.FileHandler(performance_log_path, encoding="utf-8")
console_handler = logging.StreamHandler(sys.stdout)

info_log.setLevel(logging.INFO)
error_log.setLevel(logging.ERROR)
performance_log.setLevel(logging.DEBUG)
console_handler.setLevel(logging.ERROR)

for handler in [info_log, error_log, performance_log, console_handler]:
    handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.handlers.clear()
logger.setLevel(logging.DEBUG)
logger.addHandler(info_log)
logger.addHandler(error_log)
logger.addHandler(performance_log)
logger.addHandler(console_handler)

# Suppress verbose logging from OpenAI, urllib3, and httpx (used internally)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ✅ Initialize OpenAI API Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) # The client is now in this file

# ✅ Path to embeddings file
EMBEDDINGS_FILE = str(EMBEDDINGS_PATH)

# ✅ Load and create FAISS index
try:
    faiss_index = FAISSIndexer.load_from_file(EMBEDDINGS_FILE, verbose=False)
    logging.info(f"✅ FAISS index created with {faiss_index.index.ntotal} embeddings.")
except Exception as e:
    logging.error(f"❌ Error creating FAISS index: {e}")
    exit(1)

ALL_METADATA = list(faiss_index.metadata.values())
ALL_FILENAMES = sorted({meta.get("filename") for meta in ALL_METADATA if isinstance(meta, dict) and meta.get("filename")})
RAW_SOURCE_CACHE = {}
QUERY_STOPWORDS = {
    "the", "and", "for", "with", "from", "into", "about", "this", "that", "these", "those",
    "what", "which", "when", "where", "who", "whom", "why", "how", "can", "could", "would",
    "should", "are", "is", "was", "were", "be", "been", "being", "a", "an", "of", "to", "in",
    "on", "at", "by", "as", "it", "its", "your", "you", "me", "my", "our", "we", "their",
    "according", "document", "documents", "information", "presented"
}

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def normalize_for_match(text):
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def tokenize_for_match(text):
    return [tok for tok in normalize_for_match(text).split() if len(tok) > 2]


def extract_query_terms(query):
    cleaned = query
    for phrase in extract_quoted_phrases(query):
        cleaned = cleaned.replace(f'"{phrase}"', " ")
    return {tok for tok in tokenize_for_match(cleaned) if tok not in QUERY_STOPWORDS}


def extract_quoted_phrases(query):
    return [phrase.strip() for phrase in re.findall(r'"([^"]+)"', query) if phrase.strip()]


def detect_target_filename(query):
    query_lower = query.lower()
    quoted_phrases = extract_quoted_phrases(query)
    has_doc_specific_cue = any(
        cue in query_lower
        for cue in ["according to", "from the document", "in the document", "from agard", "from this document"]
    )

    if not quoted_phrases and not has_doc_specific_cue:
        return None

    targets = quoted_phrases if quoted_phrases else [query]
    best_filename = None
    best_score = 0.0

    for filename in ALL_FILENAMES:
        filename_tokens = set(tokenize_for_match(os.path.splitext(filename)[0].replace("_", " ")))
        if not filename_tokens:
            continue

        score_for_file = 0.0
        for target in targets:
            target_tokens = set(tokenize_for_match(target))
            if not target_tokens:
                continue
            overlap = len(target_tokens & filename_tokens) / max(len(target_tokens), 1)
            score_for_file = max(score_for_file, overlap)

        if score_for_file > best_score:
            best_score = score_for_file
            best_filename = filename

    threshold = 0.50 if quoted_phrases else 0.35
    return best_filename if best_score >= threshold else None


def rank_chunks_by_lexical_overlap(chunks, query):
    query_tokens = extract_query_terms(query)
    quoted_phrases = [normalize_for_match(p) for p in extract_quoted_phrases(query)]

    ranked = []
    for chunk in chunks:
        text = chunk.get("text", "")
        if not text:
            continue
        text_norm = normalize_for_match(text)
        text_tokens = set(text_norm.split())

        overlap_score = len(query_tokens & text_tokens)
        quote_bonus = sum(4 for phrase in quoted_phrases if phrase and phrase in text_norm)
        total_score = overlap_score + quote_bonus
        ranked.append((total_score, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in ranked]


def extract_raw_text_from_source_file(filename):
    if filename in RAW_SOURCE_CACHE:
        return RAW_SOURCE_CACHE[filename]

    source_path = DOCUMENTS_DIR / filename
    if not source_path.exists():
        RAW_SOURCE_CACHE[filename] = ""
        return ""

    text = ""
    suffix = source_path.suffix.lower()

    try:
        if suffix == ".pdf":
            with pdfplumber.open(source_path) as pdf:
                text = "\n".join([(page.extract_text() or "") for page in pdf.pages])
        elif suffix == ".docx":
            doc = DocxDocument(source_path)
            text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
    except Exception as error:
        logging.warning("Failed to extract raw source text from %s: %s", source_path, error)
        text = ""

    RAW_SOURCE_CACHE[filename] = text
    return text


def split_into_passages(text, target_size=1200, overlap=200):
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) > 80]
    if len(paragraphs) >= 20:
        return paragraphs

    normalized = re.sub(r"\s+", " ", text).strip()
    if not normalized:
        return []

    passages = []
    start = 0
    while start < len(normalized):
        end = min(start + target_size, len(normalized))
        passages.append(normalized[start:end])
        if end >= len(normalized):
            break
        start = max(0, end - overlap)
    return passages


def select_top_passages_for_query(raw_text, query, max_passages=10):
    passages = split_into_passages(raw_text)
    if not passages:
        return []

    query_tokens = extract_query_terms(query)
    if not query_tokens:
        return []

    passage_tokens_list = [set(tokenize_for_match(passage)) for passage in passages]
    doc_frequency = {token: 0 for token in query_tokens}
    for tokens in passage_tokens_list:
        for token in query_tokens:
            if token in tokens:
                doc_frequency[token] += 1

    scored = []
    for idx, (passage, passage_tokens) in enumerate(zip(passages, passage_tokens_list)):
        passage_norm = normalize_for_match(passage)
        weighted_overlap = 0.0
        for token in query_tokens:
            if token in passage_tokens:
                weighted_overlap += 1.0 / (1.0 + doc_frequency[token])

        if any(token.startswith("classif") for token in query_tokens) and "classified in two areas" in passage_norm:
            weighted_overlap += 1.0
        if "measurand" in query_tokens and "measurand list" in passage_norm:
            weighted_overlap += 1.0

        total_score = weighted_overlap
        scored.append((total_score, idx, passage))

    scored.sort(key=lambda item: item[0], reverse=True)
    top = [entry for entry in scored[:max_passages] if entry[0] > 0]
    if not top:
        top = scored[:max_passages]
    return top

# ✅ Function to get embeddings
def get_embedding(text):
    """Generate embeddings for a given query."""
    try:
        response = client.embeddings.create(input=[text], model="text-embedding-ada-002")
        return response.data[0].embedding
    except OpenAIError as e:
        logging.error(f"❌ Error generating embedding: {e}")
        return None

# ✅ Function to generate a response
def generate_response(query, context, model="gpt-4-turbo", strict_mode=False, target_filename=None):
    """Generate a structured compliance-driven response using GPT-4."""
    
    strict_block = ""
    if strict_mode:
        strict_block = f"""
    DOCUMENT-GROUNDED MODE:
    - The user asked for information from a specific document.
    - Prioritize only this source: {target_filename}.
    - Answer only from the provided context snippets.
    - If information is missing in context, explicitly say it is not found.
    - Include citations in this format: [filename | chunk_id].
    - For direct facts, quote exact wording briefly when possible.
    - Do not provide generic background outside the cited snippets.
    """

    prompt = f"""
    🛠️ **Aviation Compliance Expert Analysis**
    
    You are an AI specializing in aviation safety, compliance, and regulatory risk analysis. 
    First, interact to understand the request. Read the data. Analyze the data to identify
    likely hazards and organizational gaps based on Dirty Dozen, HFACS, organizational theory
    from Mr. James Reason, or other relevant gaps associated with main Aviation Standards and
    Good practices. The analysis should have a broader organizational perspective, focusing on
    identifying likely issues within the organization's structure and processes. Responses must
    be assertive, clear, and objective, always providing evidence to support the findings.
    Communication style should be friendly and technical, ensuring clarity and professionalism
    while being approachable. The Safety Management System Advisor should ask for clarification
    if needed to ensure accurate and relevant responses. This system is strictly for improving and
    identifying internal issues to address actions correctly, enhancing the organizational safety
    culture, and contributing to the overall Safety Culture. It is forbidden to use this system
    for any other purpose. 
    Analyze the user's query and respond appropriately based on the following guidelines:

    1. For broad or open-ended questions:
       - Provide a general overview of the topic.
       - Highlight key areas of concern.
       - Suggest more specific questions for detailed analysis if needed.

    2. For specific questions about regulations, procedures, or incidents:
       - Focus on the relevant aspects without necessarily going through all analysis steps.
       - Provide direct answers backed by aviation regulations or industry standards.

    3. For questions requiring in-depth analysis:
       - Use the structured approach outlined below, adapting as necessary:
         a) Issue Analysis
         b) Regulatory Review (FAA, ICAO, EASA)
         c) Cross-Check with Accident Reports (if applicable)
         d) Risk Mitigation Framework & Safety Enhancements
         e) Compliance Validation Score & Risk Level (if appropriate)
         f) Cross-Check with Accident Investigations (if relevant)

    4. For simple factual queries:
       - Provide a concise, direct answer without extensive analysis.

    5. If the query is unclear or lacks context:
       - Ask for clarification or provide a range of possible interpretations.

    Always prioritize accuracy and relevance in your responses.
    Use the provided context first.
    If the context is insufficient, clearly state the limitation.
    Do not repeat sections, paragraphs, or bullet points.
    {strict_block}
    
    Context:
    {context}

    Here is the user's question:
    {query}

    Ensure your response is well-structured, factual, and backed by aviation regulations.
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.15 if strict_mode else 0.3,
            max_tokens=900 if strict_mode else 1200
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        logging.error(f"❌ Error calling GPT-4: {e}")
        return "I'm sorry, but I encountered an error while generating a response."


# ✅ Function to handle chat loop
def chat_loop():
    """Interactive chat loop for AviationAI."""
    print("Welcome to the AviationAI Chat System!")
    print("Type 'exit' or 'quit' to end the conversation.")

    session_metadata_file = os.path.join(chat_id, "session_metadata.json")

    # ✅ Load session metadata once (avoid redundant reloading)
    session_metadata = {}
    if os.path.exists(session_metadata_file):
        try:
            with open(session_metadata_file, "r", encoding="utf-8") as file:
                session_metadata = json.load(file)
        except json.JSONDecodeError:
            logging.error("⚠️ Corrupted session metadata file. Resetting...")
            session_metadata = {}

    # ✅ Ensure session_id is initialized
    session_id = None
    past_exchanges = []  # Initialize chat history
    chat_cache = {}  # Cache for quick retrieval

     # ✅ Allow user to select a previous session or start a new one
    if session_metadata:
        print("\n📌 Available Previous Sessions:")

        # Get the last 5 sessions only
        recent_sessions = list(session_metadata.items())[-5:]  # Keep only the last 5

        for i, (sid, title) in enumerate(recent_sessions, 1):
            print(f"{i}. {title}")

        try:
            choice = int(input("\nEnter session number to continue (or 0 for a new session): "))
            if 1 <= choice <= len(recent_sessions):  # Ensure valid choice
                session_id, session_title = recent_sessions[choice - 1]
                print(f"✅ Continuing session: {session_title}")
                past_exchanges = chat_cache.get(session_id, retrieve_chat_from_db(session_id))
                chat_cache[session_id] = past_exchanges  # Store in cache
            else:
                session_id = str(uuid.uuid4())
                print("🔄 Starting a new session...")
        except ValueError:
            print("⚠️ Invalid input, creating a new session.")
            session_id = str(uuid.uuid4())
    else:
        session_id = str(uuid.uuid4())

    # ✅ Assign a title for new sessions
    if session_id not in session_metadata:
        session_subject = input("Enter a short title for this session (e.g., 'HFACS Methodology Discussion'): ").strip()
        session_metadata[session_id] = session_subject

    # ✅ Save updated session metadata
    with open(session_metadata_file, "w", encoding="utf-8") as file:
        json.dump(session_metadata, file, indent=4)
    # ✅ Retrieve past chat history correctly

    chat_history = [(ex["user_query"], ex["ai_response"]) for ex in past_exchanges if isinstance(ex, dict) and "user_query" in ex and "ai_response" in ex]

    max_history = 5  # Keep only the last 5 exchanges in chat history

    while True:
        try:
            query = input("\nUser: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nThank you for using the Aviation RAG Chat System. Goodbye!")
            break

        if query.lower() in {"exit", "quit", "q"}:
            print("Thank you for using the Aviation RAG Chat System. Goodbye!")
            break

        if not query:
            continue

        try:
            target_filename = detect_target_filename(query)
            strict_mode = target_filename is not None

            logging.info("Generating query embedding...")
            query_embedding = get_embedding(query)
            if query_embedding is None:
                raise ValueError("Failed to generate query embedding")

            logging.info("Searching FAISS for relevant documents...")
            results = faiss_index.search(query_embedding, k=60 if strict_mode else 20)
            context_texts = []
            total_tokens = 0
            max_tokens = 3500
            max_chunks_per_file = 4
            per_file_count = {}
            seen_signatures = set()

            if strict_mode:
                logging.info(f"Document-grounded mode enabled for: {target_filename}")
                raw_text = extract_raw_text_from_source_file(target_filename)
                raw_passages = select_top_passages_for_query(raw_text, query, max_passages=12) if raw_text else []

                for _, passage_index, passage in raw_passages:
                    signature = " ".join(passage.lower().split())[:400]
                    if signature in seen_signatures:
                        continue

                    doc_tokens = num_tokens_from_string(passage)
                    if total_tokens + doc_tokens > max_tokens:
                        break

                    context_texts.append(
                        f"[SOURCE filename={target_filename}; chunk_id=raw_passage_{passage_index}]"
                        f"\n{passage}"
                    )
                    total_tokens += doc_tokens
                    seen_signatures.add(signature)

                # Fallback to preprocessed embeddings if raw source extraction is unavailable
                if not context_texts:
                    file_chunks = [meta for meta in ALL_METADATA if meta.get("filename") == target_filename]
                    ranked_chunks = rank_chunks_by_lexical_overlap(file_chunks, query)
                    if not ranked_chunks:
                        ranked_chunks = [
                            meta for meta, _ in results if meta.get("filename") == target_filename
                        ]

                    for metadata in ranked_chunks:
                        doc_text = metadata.get("text", "")
                        filename = metadata.get("filename", "unknown")
                        chunk_id = metadata.get("chunk_id", "unknown")
                        if not doc_text:
                            continue

                        signature = " ".join(doc_text.lower().split())[:400]
                        if signature in seen_signatures:
                            continue

                        doc_tokens = num_tokens_from_string(doc_text)
                        if total_tokens + doc_tokens > max_tokens:
                            break

                        context_texts.append(f"[SOURCE filename={filename}; chunk_id={chunk_id}]\n{doc_text}")
                        total_tokens += doc_tokens
                        seen_signatures.add(signature)
            else:
                for metadata, score in results:
                    doc_text = metadata.get('text', '')
                    filename = metadata.get('filename', 'unknown')
                    chunk_id = metadata.get('chunk_id', 'unknown')
                    if not doc_text:
                        continue

                    signature = " ".join(doc_text.lower().split())[:400]
                    if signature in seen_signatures:
                        continue

                    file_count = per_file_count.get(filename, 0)
                    if file_count >= max_chunks_per_file:
                        continue

                    doc_tokens = num_tokens_from_string(doc_text)
                    if total_tokens + doc_tokens > max_tokens:
                        break

                    context_texts.append(f"[SOURCE filename={filename}; chunk_id={chunk_id}]\n{doc_text}")
                    total_tokens += doc_tokens
                    seen_signatures.add(signature)
                    per_file_count[filename] = file_count + 1
                 
            context = "\n".join(context_texts)
       
            # Generate response
            response = generate_response(
                query,
                context,
                strict_mode=strict_mode,
                target_filename=target_filename,
            )
            print("\nAviationAI:", response)
            # Store chat in AstraDB
            store_chat_in_db(session_id, query, response)
            # Update chat history
            past_exchanges.append((query, response))
            if len(past_exchanges) > 5:
                past_exchanges = past_exchanges[-5:]

            chat_history.append((query, response))
            chat_history = chat_history[-max_history:]

        except Exception as e:
            logging.error(f"Error: {e}")
            print(f"An error occurred: {e}")
# ✅ Run the chat loop
if __name__ == "__main__":
    chat_loop()
