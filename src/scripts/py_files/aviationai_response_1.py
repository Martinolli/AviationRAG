import json
from openai import OpenAI
from openai import OpenAIError
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from dotenv import load_dotenv
import os
import time
from nltk.corpus import wordnet
import logging
import subprocess
import uuid
import sys
import faiss

# Load environment variables
load_dotenv()

# Define absolute paths
base_dir = r'C:\Users\Aspire5 15 i7 4G2050\ProjectRAG\AviationRAG'
log_dir = os.path.join(base_dir, 'logs')  # Define the path to the logs folder
chat_dir = os.path.join(base_dir, 'chat')  # Define the path to the chat folder
chat_id = os.path.join(base_dir, 'chat_id')  # Define the path to the chat_id folder

# Ensure the chat directory exists
if not os.path.exists(chat_dir):
    os.makedirs(chat_dir)

if not os.path.exists(chat_id):
    os.makedirs(chat_id)

# Ensure log directory exists
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Define multiple log files
info_log_path = os.path.join(log_dir, 'info.log')
error_log_path = os.path.join(log_dir, 'error.log')
performance_log_path = os.path.join(log_dir, 'performance.log')

log_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# Ensure UTF-8 encoding for all handlers
info_log = logging.FileHandler("logs/info.log", encoding="utf-8")
error_log = logging.FileHandler("logs/error.log", encoding="utf-8")
performance_log = logging.FileHandler("logs/performance.log", encoding="utf-8")
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

def store_chat_in_db(session_id, user_query, ai_response):
    """
    Calls the Node.js script to store chat in AstraDB.
    """
    # Define the correct path to store_chat.js inside src/scripts/
    
    script_path = os.path.join(os.path.dirname(__file__), '..', 'js_files', 'store_chat.js')

    if not ai_response or len(ai_response) < 10:
        logging.error("⚠️ Invalid AI response detected! Storing default message.")
        ai_response = "AI response was incomplete or not available."

    chat_data = {
        "action": "store",
        "session_id": session_id,
        "user_query": user_query,
        "ai_response": ai_response
    }
    # Call the JavaScript file with the correct path
    try:
        subprocess.run(
            ["node", script_path, json.dumps(chat_data)],
            check=True,
            cwd=os.path.join(os.path.dirname(__file__), "..", "js_files")  # Ensure correct working directory
    )

        print("Chat stored successfully in AstraDB!")
        logging.info(f"💾 Storing chat for session: {session_id} | Query: {user_query[:50]}...")

    except subprocess.CalledProcessError as e:
        logging.error(f"Error storing chat: {e}")

def retrieve_chat_from_db(session_id, limit=5):
    """
    Calls the Node.js script to retrieve chat history from AstraDB.
    """
    script_path = os.path.join(os.path.dirname(__file__), '..', 'js_files', 'store_chat.js')
    logging.info(f"📥 Retrieving chat messages for session: {session_id}")

    if not session_id.strip():
        print("⚠️ Warning: `session_id` is empty! Generating a new one...")
        session_id = str(uuid.uuid4())  # Assign a new session if empty

    chat_data = {
        "action": "retrieve",
        "session_id": session_id,
        "limit": limit
    }

    try:
        result = subprocess.run(
            ["node", script_path, json.dumps(chat_data)], 
            capture_output=True, text=True, check=True
        )
        output = result.stdout.strip()

        # Extract only the JSON part from the output
        first_brace = output.find("{")
        if first_brace != -1:
            output = output[first_brace:]  # Remove any extra log lines before JSON

        try:
            parsed_output = json.loads(output)
            if parsed_output.get("success", False):
                return parsed_output.get("messages", [])
            else:
                logging.error(f"Chat retrieval failed. Parsed output: {parsed_output}")
                return []
        except json.JSONDecodeError as e:
            logging.error(f"JSON parsing error: {e}. Raw output: {output}")
            return []

    except subprocess.CalledProcessError as e:
        logging.error(f"Error retrieving chat: {e}")
        return []

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

def create_weighted_context(top_results):
    combined_context = ""
    total_weight = sum(result['similarity'] for result in top_results)
    for result in top_results:
        weight = result['similarity'] / total_weight
        combined_context += f"{weight:.2f} * {result['text']}\n"
    return combined_context

embeddings_cache = {}  # In-memory cache for embeddings

def load_embeddings(file_path):
    """Load embeddings from a JSON file and create FAISS index."""
    global embeddings_cache

    if file_path in embeddings_cache:
        return embeddings_cache[file_path]

    if not os.path.exists(file_path):
        logging.error(f"🚨 Embeddings file not found: {file_path}")
        return None, []

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        index, embeddings = create_faiss_index(data)
        embeddings_cache[file_path] = (index, embeddings)
        return index, embeddings

    except json.JSONDecodeError as e:
        logging.error(f"🚨 Error loading embeddings JSON: {e}")
        return None, []

def create_faiss_index(embeddings):
    """Create a FAISS index from the embeddings."""
    dimension = len(embeddings[0]['embedding'])
    index = faiss.IndexFlatIP(dimension)  # Inner product is equivalent to cosine similarity for normalized vectors
    vectors = np.array([emb['embedding'] for emb in embeddings], dtype=np.float32)
    faiss.normalize_L2(vectors)  # Normalize vectors
    index.add(vectors)
    return index, embeddings

def search_faiss_index(index, query_vector, k=15):
    """Search the FAISS index for similar vectors."""
    query_vector = np.array([query_vector], dtype=np.float32)
    faiss.normalize_L2(query_vector)
    total_vectors = index.ntotal
    k = min(k, total_vectors)  # Ensure k is not larger than the total number of vectors
    logging.info(f"Searching FAISS index with {total_vectors} vectors for top {k} results")
    distances, indices = index.search(query_vector, k)
    logging.info(f"FAISS search returned {len(indices[0])} results")
    return distances[0], indices[0]

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
    max_tokens = min(500, 4000 - len(truncated_full_context.split()))

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

    # ✅ Load embeddings only once at the beginning
    try:
        print("Loading embeddings...")
        faiss_index, embeddings = load_embeddings(EMBEDDINGS_FILE)
        if faiss_index is None:
            logging.error("Failed to load embeddings and create FAISS index.")
            return
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

            print("Searching for similar embeddings...")
            distances, indices = search_faiss_index(faiss_index, query_embedding)
            logging.info(f"Got {len(indices)} results from FAISS search")

            top_results = [
                {**embeddings[i], 'similarity': float(distances[i])}
                for i in indices
                if i < len(embeddings) and distances[i] > 0.6
                ]

            logging.info(f"Filtered to {len(top_results)} results above threshold")
        
            if not top_results:
                logging.error(f"⚠️ No valid embeddings found for query: {expanded_query}")
                print(f"⚠️ No relevant data found for: {expanded_query}")
                response = "I'm sorry, but I couldn't find any relevant information to answer your query."
            else:
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

            store_chat_in_db(session_id, expanded_query, response)
            print(f"Expanded Query: {expanded_query}")

        except Exception as e:
            logging.error(f"Error in chat loop: {e}")
            print("\nAviationAI: I'm sorry, but I encountered an error while processing your query. Please try again.")

if __name__ == "__main__":
    chat_loop()