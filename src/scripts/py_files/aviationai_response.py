import json
from openai import OpenAI
from openai import OpenAIError
import numpy as np
from dotenv import load_dotenv
import os
import time
from nltk.corpus import wordnet
import logging
import uuid
import sys

from chat_db import retrieve_chat_from_db, store_chat_in_db
from config import CHAT_DIR, CHAT_ID_DIR, LOG_DIR, PROJECT_ROOT


# Load environment variables
load_dotenv(PROJECT_ROOT / ".env")

# Define paths
log_dir = LOG_DIR
chat_dir = CHAT_DIR
chat_id = CHAT_ID_DIR

# Ensure the chat directory exists
chat_dir.mkdir(parents=True, exist_ok=True)
chat_id.mkdir(parents=True, exist_ok=True)
log_dir.mkdir(parents=True, exist_ok=True)

# Define multiple log files
info_log_path = os.path.join(log_dir, 'info.log')
error_log_path = os.path.join(log_dir, 'error.log')
performance_log_path = os.path.join(log_dir, 'performance.log')

log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Ensure UTF-8 encoding for all handlers
info_log = logging.FileHandler(info_log_path, encoding="utf-8")
error_log = logging.FileHandler(error_log_path, encoding="utf-8")
performance_log = logging.FileHandler(performance_log_path, encoding="utf-8")
console_handler = logging.StreamHandler(sys.stdout)  # ✅ Stream to console

# Set logging levels
info_log.setLevel(logging.INFO)
error_log.setLevel(logging.ERROR)
performance_log.setLevel(logging.DEBUG)
console_handler.setLevel(logging.INFO)

# Apply formatters
info_log.setFormatter(log_formatter)
error_log.setFormatter(log_formatter)
performance_log.setFormatter(log_formatter)
console_handler.setFormatter(log_formatter)

# Create Logger
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(info_log)
logger.addHandler(error_log)
logger.addHandler(performance_log)
logger.addHandler(console_handler)  # ✅ Enable console logging


# Set up the OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def safe_openai_call(api_function, max_retries=3, base_delay=2):
    """
    Wrapper to handle OpenAI API calls safely with retries.
    
    Args:
        api_function (function): The OpenAI API call function.
        max_retries (int): Maximum retry attempts.
        base_delay (int): Initial delay in seconds.

    Returns:
        Response from OpenAI API or None if failure persists.
    """
    for attempt in range(max_retries):
        try:
            return api_function()
        except Exception as e:
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)  # Exponential backoff
                logging.warning(f"OpenAI API error: {e}. Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                logging.error(f"Failed after {max_retries} attempts: {e}")
                return None
        
def expand_query(query):
    """Expand the user query with aviation-specific terminology."""
    prompt = f"""
    You are an aviation expert. Given the aviation-related query below,
    expand it using technical aviation terms, acronyms, synonyms, and industry jargon.

    - **Do not change its meaning.**
    - **Do not rephrase the question.**
    - **Only append relevant aviation-related terms, acronyms, or synonyms at the end.**
    
    **Original Query:** "{query}"
    
    Provide only the modified version:
    """
  
    response = safe_openai_call(lambda: client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.4,  # Lower temperature for more deterministic output
        max_tokens=80  # Reduce max_tokens to prevent long outputs
    ))

    expanded_query = response.choices[0].message.content.strip() if response else query
    logging.info(f"🔍 Expanded Query: {expanded_query}")
    return expanded_query


def get_user_feedback():
    while True:
        feedback = input("Was this response helpful? (y/n): ").lower()
        if feedback in ['y', 'n']:
            return feedback == 'y'
        print("Please enter 'y' for yes or 'n' for no.")

embedding_cache = {}  # In-memory cache for embeddings

def get_embedding(text):
    """Generate or retrieve cached embedding for the given text."""
    global embedding_cache

    if text in embedding_cache:
        return embedding_cache[text]  # Return cached embedding

    try:
        if not text.strip():
            logging.error("⚠️ Empty text received for embedding generation!")
            return None

        response = client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )

        embedding_vector = response.data[0].embedding
        if embedding_vector and len(embedding_vector) > 10:
            embedding_cache[text] = embedding_vector  # Store in cache

        return embedding_vector

    except Exception as e:
        logging.error(f"Error generating embedding: {e}")
        return None

def get_dynamic_top_n(similarities, max_n=15, threshold=0.6):
    sorted_similarities = sorted(similarities, reverse=True)
    for i, sim in enumerate(sorted_similarities):
        if sim < threshold or i == max_n:
            return i
    return max_n

def create_weighted_context(top_results):
    combined_context = ""
    total_weight = sum(result['similarity'] for result in top_results)
    for result in top_results:
        weight = result['similarity'] / total_weight
        combined_context += f"{weight:.2f} * {result['text']}\n"
    return combined_context

embeddings_cache = {}  # In-memory cache for embeddings

def load_embeddings(file_path, batch_size=1000):
    """Load embeddings from a JSON file and cache results."""
    global embeddings_cache

    if file_path in embeddings_cache:
        return embeddings_cache[file_path]  # Return cached embeddings

    if not os.path.exists(file_path):
        logging.error(f"🚨 Embeddings file not found: {file_path}")
        return []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        embeddings_cache[file_path] = data  # Store embeddings in cache

        for i in range(0, len(data), batch_size):
            yield data[i:i + batch_size]

        return data

    except json.JSONDecodeError as e:
        logging.error(f"🚨 Error loading embeddings JSON: {e}")
        return []


def compute_cosine_similarity(vec1, vec2):
    """
    Compute cosine similarity between two vectors.
    
    Args:
        vec1 (list or np.array): First vector
        vec2 (list or np.array): Second vector
    
    Returns:
        float: Cosine similarity between vec1 and vec2
    """
    try:
        # Ensure vectors are not None
        if vec1 is None or vec2 is None:
            logging.error("One or both vectors are None")
            return 0.0
        
        # Convert to NumPy arrays
        vec1 = np.array(vec1, dtype=np.float32)
        vec2 = np.array(vec2, dtype=np.float32)

        # Compute magnitudes
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)

        # Handle cases where magnitude is zero
        if magnitude1 == 0 or magnitude2 == 0:
            logging.error("One or both vectors have zero magnitude, returning similarity as 0.")
            return 0.0

        # Compute cosine similarity
        similarity = np.dot(vec1, vec2) / (magnitude1 * magnitude2)
        return similarity

    except Exception as e:
        logging.error(f"Error in compute_cosine_similarity: {str(e)}")
        return 0.0

def filter_and_rank_embeddings(embeddings, similarities, top_n=15, min_similarity=0.7):
    """Filter and rank embeddings based on similarity scores."""
    # Calculate the average similarity
    avg_similarity = np.mean(similarities)
    # Use max of average similarity and min_similarity as the threshold
    threshold = max(avg_similarity, min_similarity)
    
    return sorted(
        [
            {**emb, 'similarity': sim}
            for emb, sim in zip(embeddings, similarities)
            if sim > threshold and isinstance(emb, dict) and 'embedding' in emb
        ],
        key=lambda x: x['similarity'],
        reverse=True
    )[:top_n]

def generate_response(context, expanded_query, full_context, model):
    """Generate a response using OpenAI."""
    max_context_length = 8000  # Adjust this value based on your needs
    max_retries = 3
    
    # Implement a sliding window for chat history
    max_past_messages = 3  # Keep only the last 3 exchanges
    chat_history = full_context.split("\n\n")[-max_past_messages:]
    truncated_full_context = "\n\n".join(chat_history[-3:])  # Keep only the last 3 exchanges

    # Truncate the context if it's too long
    if len(truncated_full_context) > max_context_length:
        truncated_full_context = truncated_full_context[-max_context_length:]
    
    logging.info(f"🤖 Calling GPT-4 for expanded query: {expanded_query[:50]}...")
    
    prompt = f"""
    You are an AI assistant specializing in aviation. Provide detailed, thorough answers with examples
    where relevant. Use the context and history below to answer the user's question:
    
    Chat History and Full Context:
    {truncated_full_context}

    Context:
    {context}

    Human: {expanded_query}

    Provide a detailed, comprehensive, and accurate response based on the context above. 
    Include relevant facts, explanations, and examples where appropriate. 
    For each key piece of information in your response, cite the source document in square brackets, 
    e.g., [Document: Safety Manual]. If information comes from multiple sources, list all relevant sources.
    If the context doesn't contain enough information to fully answer the question, 
    clearly state what information is missing or uncertain.
    """
    # Calculate max tokens dynamically
    max_tokens = min(500, 8000 - len(truncated_full_context.split()))

    import random

    for attempt in range(max_retries):
        try:
            start_time = time.time() # Start time tracking
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=max_tokens
            )
            execution_time = time.time() - start_time  # Calculate execution time
            logging.debug(f"⚡ GPT-4 Response Time: {execution_time:.2f} sec")
            response_text = response.choices[0].message.content.strip()
            logging.info(f"✅ GPT-4 Response Generated: {response_text[:50]}...")  # Show first 50 characters

            if not response_text:
                logging.error("⚠️ GPT-4 returned an empty response!")
                return "I'm sorry, but I couldn't generate a response."
            
            return response_text
        
        except OpenAIError as e:
            if "429 Too Many Requests" in str(e):
                wait_time = (10 * (attempt + 1)) + random.uniform(1, 5)  # Adds random delay to prevent API bans
                print(f"⚠️ OpenAI rate limit exceeded. Retrying in {wait_time:.2f} seconds...")
                time.sleep(wait_time)
            else:
                logging.error(f"❌ Error calling GPT-4: {e}")
                return "I'm sorry, but I encountered an error while generating a response."

def chat_loop():
    """Main chat loop for AviationAI"""
    
    EMBEDDINGS_FILE = "data/embeddings/aviation_embeddings.json"
    MODEL = "gpt-4"  # Change between "gpt-3.5-turbo" or "gpt-4"
    
    print("Welcome to the AviationAI Chat System!")
    print("Type 'exit' to end the conversation.")

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
        for i, (sid, title) in enumerate(session_metadata.items(), 1):
            print(f"{i}. {title} (Session ID: {sid[:8]}...)")

        try:
            choice = int(input("\nEnter session number to continue (or 0 for a new session): "))
            if 1 <= choice <= len(session_metadata):
                session_id = list(session_metadata.keys())[choice - 1]
                print(f"✅ Continuing session: {session_metadata[session_id]}")
                past_exchanges = chat_cache.get(
                    session_id,
                    retrieve_chat_from_db(session_id, warn_on_empty_session=True),
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
        session_subject = input("Enter a short title for this session (e.g., 'HFACS Methodology Discussion'): ").strip()
        session_metadata[session_id] = session_subject

    # ✅ Save updated session metadata
    with open(session_metadata_file, "w", encoding="utf-8") as file:
        json.dump(session_metadata, file, indent=4)

    # ✅ Load embeddings only once at the beginning
    try:
        print("Loading embeddings...")
        embeddings = load_embeddings(EMBEDDINGS_FILE)
    except Exception as e:
        logging.error(f"Error loading embeddings: {e}")
        return
    
    # ✅ Retrieve past chat history correctly
    chat_history = [(ex["user_query"], ex["ai_response"]) for ex in past_exchanges if isinstance(ex, dict) and "user_query" in ex and "ai_response" in ex]

    max_history = 5  # Keep only the last 5 exchanges in chat history

    while True:
        QUERY_TEXT = input("\nUser: ")
        if QUERY_TEXT.lower() == 'exit':
            print("Thank you for using the AviationAI Chat System. Goodbye!")
            break
        
        try:
            print("Generating query embedding...")
            expanded_query = expand_query(QUERY_TEXT)
            query_embedding = get_embedding(expanded_query)
            if query_embedding is None:
                logging.error(f"❌ Failed to generate query embedding for: {expanded_query}")
                print("⚠️ Embedding generation failed! Cannot process the query.")
                continue

            print("Processing embeddings...")
            top_results = []

            for batch in load_embeddings(EMBEDDINGS_FILE):  
                valid_embeddings = [emb for emb in batch if isinstance(emb, dict) and 'embedding' in emb and len(emb['embedding']) > 15]

                if valid_embeddings:
                    print(f"✅ Processing {len(valid_embeddings)} embeddings in batch...")
                    similarities = [compute_cosine_similarity(query_embedding, emb['embedding']) for emb in valid_embeddings]
                    top_results.extend(filter_and_rank_embeddings(valid_embeddings, similarities, top_n=15))  
                else:
                    logging.warning("⚠️ No valid embeddings found in this batch. Skipping.")
                    continue

            if not top_results:
                logging.error(f"⚠️ No valid embeddings found for query: {expanded_query}")
                print(f"⚠️ No relevant data found for: {expanded_query}")
                response = "I'm sorry, but I couldn't find any relevant information to answer your query."
            else:
                dynamic_top_n = get_dynamic_top_n(similarities)
                top_results = filter_and_rank_embeddings(top_results, similarities, top_n=dynamic_top_n)

                combined_context = create_weighted_context(top_results)
                chat_context = "\n".join([f"Human: {q}\nAI: {a}" for q, a in chat_history])
                full_context = f"{chat_context}\n\n{combined_context}"

                print("Generating response...")
                logging.info("🛠️ Calling GPT-4 to generate response...")
                response = generate_response(combined_context, expanded_query, full_context, MODEL)

                if response and len(response) >= 10:
                    logging.info(f"✅ GPT-4 Response Generated: {response[:50]}...")
                else:
                    logging.error("⚠️ GPT-4 returned an empty or invalid response!")
                    response = "I'm sorry, but I couldn't generate a meaningful response. Please try rephrasing your query."

            print("\nAviationAI:", response)

            is_helpful = get_user_feedback()
            if not is_helpful:
                print("I'm sorry the response wasn't helpful. Let me try to improve.")

            chat_history.append((expanded_query, response))
            chat_history = chat_history[-max_history:]

            store_chat_in_db(session_id, expanded_query, response, print_success=True, log_success=True)
            print(f"Expanded Query: {expanded_query}")

        except Exception as e:
            logging.error(f"Error in chat loop: {e}")
            print("\nAviationAI: I'm sorry, but I encountered an error while processing your query. Please try again.")

if __name__ == "__main__":
    chat_loop()
