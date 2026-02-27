"""
Aviation RAG (Retrieval Augmented Generation) Engine.

This module implements a comprehensive Retrieval Augmented Generation system for aviation-related
queries. It combines OpenAI's language models with a FAISS vector index to retrieve relevant
aviation documents and provide accurate, context-aware responses.

Key Features:
    - Vector similarity search using FAISS index for efficient document retrieval
    - Multi-format document processing (PDF, DOCX) with text extraction and chunking
    - Query understanding and normalization with stopword filtering
    - Lexical overlap ranking for improved result relevance
    - OpenAI API integration for conversational responses
    - Session-based chat history management with persistent storage
    - Comprehensive logging with separate handlers for info, errors, and performance metrics
    - Token counting and context window management using tiktoken

Main Components:
    - Query processing: normalize, tokenize, and extract terms from user queries
    - Document retrieval: rank and select relevant chunks using FAISS and lexical overlap
    - Response generation: use OpenAI API with context-aware prompts
    - Chat management: store and retrieve conversation history
    - Document handling: extract text from PDFs and DOCX files with caching

Module supports aviation-specific keyword detection (accidents, incidents, regulatory requirements)
and document-specific query routing to improve retrieval accuracy.
"""

import json
import logging
import os
import re
import sys
import unicodedata
import uuid

from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from faiss_indexer import FAISSIndexer
import tiktoken
import pdfplumber
from docx import Document as DocxDocument

from chat_db import retrieve_chat_from_db, store_chat_in_db
from config import (
    CHAT_DIR,
    CHAT_ID_DIR,
    DOCUMENTS_DIR,
    EMBEDDINGS_FILE as EMBEDDINGS_PATH,
    LOG_DIR,
    PROJECT_ROOT,
)

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

log_formatter = logging.Formatter(
    "%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
)

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
client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)  # The client is now in this file

# ✅ Path to embeddings file
EMBEDDINGS_FILE = str(EMBEDDINGS_PATH)

# ✅ Load and create FAISS index
try:
    faiss_index = FAISSIndexer.load_from_file(EMBEDDINGS_FILE, verbose=False)
    logging.info("✅ FAISS index created with %d embeddings.", faiss_index.index.ntotal)
except (FileNotFoundError, ValueError, OSError) as e:
    logging.error("❌ Error creating FAISS index: %s", e)
    exit(1)

ALL_METADATA = list(faiss_index.metadata.values())
ALL_FILENAMES = sorted(
    {
        meta.get("filename")
        for meta in ALL_METADATA
        if isinstance(meta, dict) and meta.get("filename")
    }
)
RAW_SOURCE_CACHE = {}
QUERY_STOPWORDS = {
    "the",
    "and",
    "for",
    "with",
    "from",
    "into",
    "about",
    "this",
    "that",
    "these",
    "those",
    "what",
    "which",
    "when",
    "where",
    "who",
    "whom",
    "why",
    "how",
    "can",
    "could",
    "would",
    "should",
    "are",
    "is",
    "was",
    "were",
    "be",
    "been",
    "being",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "at",
    "by",
    "as",
    "it",
    "its",
    "your",
    "you",
    "me",
    "my",
    "our",
    "we",
    "their",
    "according",
    "document",
    "documents",
    "information",
    "presented",
}
SOURCE_TAG_PATTERN = re.compile(r"\[SOURCE filename=(.*?); chunk_id=(.*?)\]")
ACCIDENT_KEYWORDS = {
    "accident",
    "incident",
    "crash",
    "event",
    "fatality",
    "ntsb",
    "aaib",
    "investigation",
}
REGULATORY_KEYWORDS = {
    "part",
    "cfr",
    "cs",
    "requirement",
    "requirements",
    "compliance",
    "certification",
    "faa",
    "easa",
    "icao",
    "order",
    "advisory",
    "ac",
    "standard",
    "regulation",
    "airworthiness",
}


def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    """
    Count the number of tokens in a string using the specified encoding.

    Args:
        string: The text to count tokens for.
        encoding_name: The tiktoken encoding to use. Defaults to "cl100k_base" (GPT-4).

    Returns:
        The number of tokens in the string.
    """
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


def normalize_for_match(text):
    """
    Normalize text for matching by converting to lowercase and removing non-alphanumeric characters.

    Args:
        text: The text to normalize.

    Returns:
        Normalized text with only lowercase alphanumeric characters and spaces.
    """
    return re.sub(r"[^a-z0-9]+", " ", str(text).lower()).strip()


def tokenize_for_match(text):
    """
    Tokenize normalized text, filtering out tokens with 2 or fewer characters.

    Args:
        text: The text to tokenize.

    Returns:
        List of tokens with length > 2 from the normalized text.
    """
    return [tok for tok in normalize_for_match(text).split() if len(tok) > 2]


def extract_query_terms(query):
    """
    Extract meaningful terms from a user query by removing quoted phrases and stopwords.

    Args:
        query: The user's search query.

    Returns:
        Set of meaningful terms from the query after removing quoted phrases and stopwords.
    """
    cleaned = query
    for phrase in extract_quoted_phrases(query):
        cleaned = cleaned.replace(f'"{phrase}"', " ")
    return {tok for tok in tokenize_for_match(cleaned) if tok not in QUERY_STOPWORDS}


def normalize_output_text(text):
    """
    Normalize output text by fixing encoding issues and standardizing quotation/dash characters.

    Args:
        text: The text to normalize.

    Returns:
        Normalized text with corrected Unicode characters and cleaned whitespace.
    """
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    # Replace common broken glyphs seen from mixed encodings.
    value = value.replace("�", "'")
    value = value.replace("\u2018", "'").replace("\u2019", "'")
    value = value.replace("\u201c", '"').replace("\u201d", '"')
    value = value.replace("\u2013", "-").replace("\u2014", "-")
    value = re.sub(r"[ \t]+", " ", value)
    return value.strip()


def extract_quoted_phrases(query):
    """
    Extract phrases enclosed in double quotes from a query string.

    Args:
        query: The query string to search for quoted phrases.

    Returns:
        List of quoted phrases (without the quotes) found in the query.
    """
    return [
        phrase.strip() for phrase in re.findall(r'"([^"]+)"', query) if phrase.strip()
    ]


def detect_target_filename(query):
    """
    Detect if a user query references a specific document and return its filename.

    This function looks for quoted phrases or document-specific cues in the query to
    identify if the user is asking about a particular document. It uses token overlap
    matching to find the most likely document.

    Args:
        query: The user's query string.

    Returns:
        The filename of the detected document, or None if no specific document is referenced.
    """
    query_lower = query.lower()
    quoted_phrases = extract_quoted_phrases(query)
    has_doc_specific_cue = any(
        cue in query_lower
        for cue in [
            "according to",
            "from the document",
            "in the document",
            "from agard",
            "from this document",
        ]
    )

    if not quoted_phrases and not has_doc_specific_cue:
        return None

    targets = quoted_phrases if quoted_phrases else [query]
    best_filename = None
    best_score = 0.0

    for filename in ALL_FILENAMES:
        filename_tokens = set(
            tokenize_for_match(os.path.splitext(filename)[0].replace("_", " "))
        )
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
    """
    Rank document chunks by lexical overlap with query terms.

    Scores chunks based on how many query terms they contain, with bonus points for
    exact phrase matches. Returns chunks sorted by relevance score (highest first).

    Args:
        chunks: List of document chunks (each as a dict with 'text' key).
        query: The user's search query.

    Returns:
        Chunks sorted by lexical overlap score in descending order.
    """
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
        quote_bonus = sum(
            4 for phrase in quoted_phrases if phrase and phrase in text_norm
        )
        total_score = overlap_score + quote_bonus
        ranked.append((total_score, chunk))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in ranked]


def extract_raw_text_from_source_file(filename):
    """
    Extract raw text from a source document (PDF or DOCX) with caching.

    Supports PDF and DOCX file formats. Results are cached to avoid repeated
    file I/O operations.

    Args:
        filename: The name of the file to extract text from.

    Returns:
        The extracted text content from the file, or empty string if extraction fails.
    """
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
    except (OSError, ValueError, IOError) as error:
        logging.warning(
            "Failed to extract raw source text from %s: %s", source_path, error
        )
        text = ""

    RAW_SOURCE_CACHE[filename] = text
    return text


def split_into_passages(text, target_size=1200, overlap=200):
    """
    Split text into passages of target size with optional overlap for context.

    First attempts to split by paragraphs if there are enough. Otherwise uses
    fixed-size chunking with overlap to handle diverse document structures.

    Args:
        text: The text to split into passages.
        target_size: Target character size for each passage. Defaults to 1200.
        overlap: Number of characters to overlap between passages. Defaults to 200.

    Returns:
        List of text passages.
    """
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
    """
    Select the top passages from raw text that best match a query.

    Uses TF-IDF-like scoring based on term frequency and inverse document frequency
    to rank passages by relevance to the query.

    Args:
        raw_text: The raw text to select passages from.
        query: The user's search query.
        max_passages: Maximum number of passages to return. Defaults to 10.

    Returns:
        List of (score, index, passage) tuples sorted by relevance score.
    """
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

        if (
            any(token.startswith("classif") for token in query_tokens)
            and "classified in two areas" in passage_norm
        ):
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


def classify_query_intent(query):
    """
    Classify the intent of a user query (accident-focused, regulatory-focused, or general).

    Determines whether a query is asking about accident reports, regulatory requirements,
    or general information to optimize document source selection.

    Args:
        query: The user's search query.

    Returns:
        Dict with boolean flags: is_accident_query and is_regulatory_query.
    """
    tokens = extract_query_terms(query)
    query_lower = query.lower()

    is_accident = bool(tokens & ACCIDENT_KEYWORDS) or any(
        key in query_lower
        for key in ["accident report", "incident report", "ntsb", "aaib"]
    )
    is_regulatory = bool(tokens & REGULATORY_KEYWORDS) or any(
        key in query_lower
        for key in ["part ", "14cfr", "cs-23", "cs-25", "faa", "easa", "icao"]
    )

    return {
        "is_accident_query": is_accident,
        "is_regulatory_query": is_regulatory,
    }


def source_kind(filename):
    """
    Classify a document filename by its document type.

    Determines the category of a source document (accident report, regulation, etc.)
    based on filename patterns, used for prioritizing retrieval results.

    Args:
        filename: The name of the document file.

    Returns:
        String indicating document kind: 'accident_report', 'regulation', 'advisory',
        'standard', 'guide', 'report', 'reference', or 'other'.
    """
    name = (filename or "").lower()
    if "accident_report" in name:
        return "accident_report"
    if "regulation_" in name:
        return "regulation"
    if "advisory_circular_" in name:
        return "advisory"
    if "standard_" in name or "sarp_" in name:
        return "standard"
    if "guide_" in name:
        return "guide"
    if "report_" in name:
        return "report"
    if "book_" in name or "paper_" in name:
        return "reference"
    if name.endswith(".pdf"):
        return "reference"
    return "other"


def source_priority_for_intent(filename, intent):
    """
    Assign a priority score to a source document based on query intent.

    Higher scores indicate documents more relevant to the user's query intent.
    Prioritization changes based on whether the query is accident-focused,
    regulatory-focused, or general.

    Args:
        filename: The name of the source document.
        intent: Dict with is_accident_query and is_regulatory_query boolean flags.

    Returns:
        Integer priority score (0-100).
    """
    kind = source_kind(filename)
    is_accident_query = intent["is_accident_query"]
    is_regulatory_query = intent["is_regulatory_query"]

    if is_accident_query:
        if kind == "accident_report":
            return 100
        if kind in {"regulation", "advisory", "standard", "guide", "report"}:
            return 70
        return 50

    if is_regulatory_query:
        if kind in {"regulation", "advisory", "standard", "guide"}:
            return 100
        if kind in {"report", "reference"}:
            return 75
        if kind == "accident_report":
            return 10
        return 50

    # Conceptual/non-accident default: prefer foundational references, down-rank accident reports.
    if kind in {"reference", "guide", "standard"}:
        return 95
    if kind in {"regulation", "advisory", "report"}:
        return 80
    if kind == "accident_report":
        return 5
    return 50


def should_skip_source_for_intent(filename, intent):
    """
    Determine if a source document should be skipped based on query intent.

    Filters out accident reports when the query is not specifically about accidents,
    to avoid returning irrelevant incident data for general queries.

    Args:
        filename: The name of the source document.
        intent: Dict with is_accident_query and is_regulatory_query boolean flags.

    Returns:
        True if the source should be skipped for this query intent, False otherwise.
    """
    kind = source_kind(filename)
    # Unless explicitly accident-focused, avoid accident report chunks in default retrieval.
    if kind == "accident_report" and not intent["is_accident_query"]:
        return True
    return False


def get_embedding(text):
    """
    Generate vector embeddings for text using OpenAI's embedding model.

    Uses the text-embedding-ada-002 model to create dense vector representations
    of text for semantic similarity search.

    Args:
        text: The text to generate embeddings for.

    Returns:
        List of floats representing the embedding vector, or None if generation fails.
    """
    try:
        response = client.embeddings.create(
            input=[text], model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except OpenAIError as e:
        logging.error("❌ Error generating embedding: %s", e)
        return None


def generate_response(
    query, context, model="gpt-4o", strict_mode=False, target_filename=None
):
    """
    Generate a compliance-driven aviation safety response using OpenAI's chat model.

    Creates a structured response that analyzes queries from an aviation safety and
    regulatory compliance perspective, using HFACS, Dirty Dozen, and organizational
    theory frameworks.

    Args:
        query: The user's question or query.
        context: Relevant document context retrieved from the knowledge base.
        model: The OpenAI model to use. Defaults to "gpt-4-turbo".
        strict_mode: If True, constrains response to only cite provided context.
                    Defaults to False.
        target_filename: When strict_mode is True, the specific document being queried.

    Returns:
        String containing the generated response, or error message if generation fails.
    """

    strict_block = ""
    if strict_mode:
        strict_block = f"""
    DOCUMENT-GROUNDED MODE:
    - Prioritize only this source: {target_filename}.
    - Answer only from the provided context snippets.
    - If information is missing in context, explicitly say it is not found.
    - Include citations in this format: [filename | chunk_id].
    - IMPORTANT: Every factual paragraph MUST include at least one citation.
    - Do not cite anything that is not present in the context.
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
            max_tokens=900 if strict_mode else 1200,
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        logging.error("❌ Error calling GPT-4: %s", e)
        return "I'm sorry, but I encountered an error while generating a response."


def format_context_entries(entries):
    """
    Format retrieved context entries with source citations for the LLM.

    Combines multiple document chunks with their source information into a
    formatted context string suitable for inclusion in LLM prompts.

    Args:
        entries: List of context entry dicts with 'filename', 'chunk_id', and 'text' keys.

    Returns:
        Formatted string with source tags and text content.
    """
    return "\n".join(
        f"[SOURCE filename={entry['filename']}; chunk_id={entry['chunk_id']}]\n{entry['text']}"
        for entry in entries
    )


def extract_citations_from_context(context):
    """
    Extract source citation information from formatted context strings.

    Parses the SOURCE tags embedded in context strings to extract document
    filenames and chunk IDs, removing duplicates.

    Args:
        context: Formatted context string with SOURCE tags.

    Returns:
        List of citation dicts with 'filename' and 'chunk_id' keys.
    """
    citations = []
    seen = set()
    for match in SOURCE_TAG_PATTERN.finditer(context):
        filename = match.group(1).strip()
        chunk_id = match.group(2).strip()
        key = (filename, chunk_id)
        if key in seen:
            continue
        seen.add(key)
        citations.append({"filename": filename, "chunk_id": chunk_id})
    return citations


# --- Citation gating utilities (STRICT MODE) ---

ANSWER_CITATION_PATTERN = re.compile(r"\[([^\[\]\|]+?)\s*\|\s*([^\[\]]+?)\]")


def extract_citations_from_answer(answer_text: str):
    """
    Extract citations from the assistant answer text.

    Parses citations in the format [filename | chunk_id] from the generated answer.

    Args:
        answer_text: The answer text to extract citations from.

    Returns:
        List of citation dicts with 'filename' and 'chunk_id' keys.
    """
    citations = []
    seen = set()

    text = str(answer_text or "")
    for match in ANSWER_CITATION_PATTERN.finditer(text):
        filename = match.group(1).strip()
        chunk_id = match.group(2).strip()
        if not filename or not chunk_id:
            continue
        key = (filename, chunk_id)
        if key in seen:
            continue
        seen.add(key)
        citations.append({"filename": filename, "chunk_id": chunk_id})
    return citations


def validate_answer_citations(answer_citations, context_entries):
    """
    Ensure citations in the answer correspond to retrieved sources.

    Validates that all citations in the answer match entries from the retrieved
    context, separating valid and invalid citations.

    Args:
        answer_citations: List of citations extracted from the answer.
        context_entries: List of context entry dicts from retrieval.

    Returns:
        Tuple of (valid_citations, invalid_citations).
    """
    allowed = set()
    for entry in context_entries or []:
        filename = str(entry.get("filename", "")).strip()
        chunk_id = str(entry.get("chunk_id", "")).strip()
        if filename and chunk_id:
            allowed.add((filename, chunk_id))

    valid = []
    invalid = []
    for c in answer_citations or []:
        filename = str(c.get("filename", "")).strip()
        chunk_id = str(c.get("chunk_id", "")).strip()
        if (filename, chunk_id) in allowed:
            valid.append({"filename": filename, "chunk_id": chunk_id})
        else:
            invalid.append({"filename": filename, "chunk_id": chunk_id})
    return valid, invalid


def apply_strict_citation_gate(answer_text: str, strict_mode: bool, context_entries):
    """
    Enforce strict citation requirements for document-grounded queries.

    In strict mode, ensures that:
      - Answer contains at least one valid citation
      - All citations reference retrieved context entries
    If validation fails, returns a safe fallback answer.

    Args:
        answer_text: The generated answer text.
        strict_mode: Whether strict citation validation is enabled.
        context_entries: List of retrieved context entries to validate against.

    Returns:
        Tuple of (answer_text, valid_citations).
        If strict mode fails, returns fallback message and empty citations list.
    """
    normalized = normalize_output_text(answer_text or "")
    answer_citations = extract_citations_from_answer(normalized)
    valid, invalid = validate_answer_citations(answer_citations, context_entries)

    if not strict_mode:
        # NON-STRICT: always show retrieved citations in the UI
        return normalized, valid

    # STRICT MODE gating:
    # 1) must cite something
    if len(valid) == 0:
        logging.warning(
            "Strict citation gate: answer had no valid citations; returning fallback."
        )
        return "Not found in provided sources.", []

    # 2) must not cite non-provided sources
    if len(invalid) > 0:
        logging.warning(
            "Strict citation gate: answer cited unknown sources: %s; returning fallback.",
            invalid,
        )
        return "Not found in provided sources.", []

    return normalized, valid


def build_retrieval_context(
    query, query_embedding, strict_mode=False, target_filename=None
):
    """
    Build retrieval context by combining vector search and lexical ranking.

    In strict mode, focuses on a specific target document. In normal mode,
    prioritizes sources based on query intent (accident, regulatory, or general).
    Combines FAISS semantic search with lexical overlap ranking and respects
    token limits for LLM context windows.

    Args:
        query: The user's search query.
        query_embedding: Vector embedding of the query from get_embedding().
        strict_mode: If True, retrieves only from target_filename. Defaults to False.
        target_filename: Specific document to focus on in strict mode.

    Returns:
        Tuple of (formatted_context_string, context_entries_list).
    """
    results = faiss_index.search(query_embedding, k=60 if strict_mode else 20)
    context_entries = []
    total_tokens = 0
    max_tokens = 3500
    max_chunks_per_file = 4
    per_file_count = {}
    seen_signatures = set()

    if strict_mode and target_filename:
        logging.info("Document-grounded mode enabled for: %s", target_filename)
        raw_text = extract_raw_text_from_source_file(target_filename)
        raw_passages = (
            select_top_passages_for_query(raw_text, query, max_passages=12)
            if raw_text
            else []
        )

        for _, passage_index, passage in raw_passages:
            signature = " ".join(passage.lower().split())[:400]
            if signature in seen_signatures:
                continue

            doc_tokens = num_tokens_from_string(passage)
            if total_tokens + doc_tokens > max_tokens:
                break

            context_entries.append(
                {
                    "filename": target_filename,
                    "chunk_id": f"raw_passage_{passage_index}",
                    "text": passage,
                }
            )
            total_tokens += doc_tokens
            seen_signatures.add(signature)

        # Fallback to preprocessed embeddings if raw source extraction is unavailable
        if not context_entries:
            file_chunks = [
                meta for meta in ALL_METADATA if meta.get("filename") == target_filename
            ]
            ranked_chunks = rank_chunks_by_lexical_overlap(file_chunks, query)
            if not ranked_chunks:
                ranked_chunks = [
                    meta
                    for meta, _ in results
                    if meta.get("filename") == target_filename
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

                context_entries.append(
                    {"filename": filename, "chunk_id": chunk_id, "text": doc_text}
                )
                total_tokens += doc_tokens
                seen_signatures.add(signature)
    else:
        intent = classify_query_intent(query)
        ranked_results = sorted(
            enumerate(results),
            key=lambda item: (
                source_priority_for_intent(item[1][0].get("filename", ""), intent),
                -item[0],  # keep earlier semantic order within same priority
            ),
            reverse=True,
        )

        for _, (metadata, _) in ranked_results:
            doc_text = metadata.get("text", "")
            filename = metadata.get("filename", "unknown")
            chunk_id = metadata.get("chunk_id", "unknown")
            if not doc_text:
                continue
            if should_skip_source_for_intent(filename, intent):
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

            context_entries.append(
                {"filename": filename, "chunk_id": chunk_id, "text": doc_text}
            )
            total_tokens += doc_tokens
            seen_signatures.add(signature)
            per_file_count[filename] = file_count + 1

    context = format_context_entries(context_entries)
    return context, context_entries


def answer_query(query, model="gpt-4o", strict_mode=None, target_filename=None):
    """
    Answer a user query using retrieval-augmented generation.

    The main entry point for the RAG system. Processes a user query by generating
    embeddings, retrieving relevant documents, and generating a compliance-focused
    response with citations.

    Args:
        query: The user's question.
        model: OpenAI model to use. Defaults to "gpt-4-turbo".
        strict_mode: Force document-grounded mode. If None, auto-detect based on query.
        target_filename: Specific document to query in strict mode.

    Returns:
        Dict with keys: 'answer' (response text), 'strict_mode' (bool), 'target_filename',
        'citations' (list of source dicts), and 'sources' (full context entries).

    Raises:
        ValueError: If query is empty or embedding generation fails.
    """
    if not query or not str(query).strip():
        raise ValueError("Query cannot be empty.")

    resolved_target = target_filename or detect_target_filename(query)
    if strict_mode is None:
        resolved_strict_mode = resolved_target is not None
    else:
        resolved_strict_mode = bool(strict_mode)

    logging.info("Generating query embedding...")
    query_embedding = get_embedding(query)
    if query_embedding is None:
        raise ValueError("Failed to generate query embedding")

    logging.info("Searching FAISS for relevant documents...")
    context, context_entries = build_retrieval_context(
        query,
        query_embedding,
        strict_mode=resolved_strict_mode,
        target_filename=resolved_target,
    )

    raw_answer = generate_response(
        query,
        context,
        model=model,
        strict_mode=resolved_strict_mode,
        target_filename=resolved_target,
    )

    normalized_answer = normalize_output_text(raw_answer or "")

    # Citations the model *actually* used (if any)
    answer_citations = extract_citations_from_answer(normalized_answer)
    valid_answer_citations, invalid_answer_citations = validate_answer_citations(
        answer_citations,
        context_entries,
    )

    if resolved_strict_mode:
        # STRICT: must cite and must be valid
        if len(valid_answer_citations) == 0 or len(invalid_answer_citations) > 0:
            logging.warning(
                "Strict citation gate failed. valid=%s invalid=%s",
                valid_answer_citations,
                invalid_answer_citations,
            )
            return {
                "answer": "Not found in provided sources.",
                "strict_mode": resolved_strict_mode,
                "target_filename": resolved_target,
                "citations": [],
                "sources": context_entries,
            }

        return {
            "answer": normalized_answer,
            "strict_mode": resolved_strict_mode,
            "target_filename": resolved_target,
            "citations": valid_answer_citations,
            "sources": context_entries,
        }

    # NON-STRICT: prefer answer citations if present, else fall back to context citations
    citations_for_ui = (
        valid_answer_citations
        if len(valid_answer_citations) > 0 and len(invalid_answer_citations) == 0
        else extract_citations_from_context(context)
    )

    return {
        "answer": normalized_answer,
        "strict_mode": resolved_strict_mode,
        "target_filename": resolved_target,
        "citations": citations_for_ui,
        "sources": context_entries,
    }


def chat_loop():
    """
    Run an interactive chat session for AviationAI.

    Manages a persistent chat session allowing users to ask multiple questions
    and maintain conversation context. Sessions are saved and can be resumed.
    Supports loading previous sessions or starting new conversations.

    Returns:
        None. Runs until user types 'exit' or 'quit'.
    """
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

        for i, (_, title) in enumerate(recent_sessions, 1):
            print(f"{i}. {title}")

        try:
            choice = int(
                input("\nEnter session number to continue (or 0 for a new session): ")
            )
            if 1 <= choice <= len(recent_sessions):  # Ensure valid choice
                session_id, session_title = recent_sessions[choice - 1]
                print(f"✅ Continuing session: {session_title}")
                past_exchanges = chat_cache.get(
                    session_id, retrieve_chat_from_db(session_id)
                )
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
        session_subject = input(
            "Enter a short title for this session (e.g., 'HFACS Methodology Discussion'): "
        ).strip()
        session_metadata[session_id] = session_subject

    # ✅ Save updated session metadata
    with open(session_metadata_file, "w", encoding="utf-8") as file:
        json.dump(session_metadata, file, indent=4)
    # ✅ Retrieve past chat history correctly

    chat_history = [
        (ex["user_query"], ex["ai_response"])
        for ex in past_exchanges
        if isinstance(ex, dict) and "user_query" in ex and "ai_response" in ex
    ]

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
            query_result = answer_query(query)
            response = query_result["answer"]
            citations = query_result.get("citations", [])
            strict_mode = query_result.get("strict_mode", False)
            target_filename = query_result.get("target_filename")

            print("\nAviationAI:", response)

            # Display citations if available
            if citations:
                print("\n📚 Sources:")
                for idx, citation in enumerate(citations, 1):
                    filename = citation.get("filename", "unknown")
                    chunk_id = citation.get("chunk_id", "unknown")
                    print(f"  {idx}. [{filename} | {chunk_id}]")

            # Display mode information
            if strict_mode and target_filename:
                print(f"\n🔒 Document-grounded mode: {target_filename}")

            # Store chat in AstraDB
            store_chat_in_db(session_id, query, response)
            # Update chat history
            past_exchanges.append((query, response))
            if len(past_exchanges) > 5:
                past_exchanges = past_exchanges[-5:]

            chat_history.append((query, response))
            chat_history = chat_history[-max_history:]

        except (ValueError, OpenAIError, KeyError) as e:
            logging.error("Error: %s", e)
            print(f"An error occurred: {e}")


# ✅ Run the chat loop
if __name__ == "__main__":
    chat_loop()
