import json
from openai import OpenAI
from openai import OpenAIError
import numpy as np
from dotenv import load_dotenv
import os
import time
from nltk.corpus import wordnet
import logging
import subprocess
import uuid


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

# Set up logging
log_file_path = os.path.join(log_dir, 'chat_system.log')
logging.basicConfig(
    filename=log_file_path,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

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
    except subprocess.CalledProcessError as e:
        logging.error(f"Error storing chat: {e}")

def retrieve_chat_from_db(session_id, limit=5):
    """
    Calls the Node.js script to retrieve chat history from AstraDB.
    """
    script_path = os.path.join(os.path.dirname(__file__), '..', 'js_files', 'store_chat.js')

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
        # Convert input to numpy arrays if they're not already
        vec1 = np.asarray(vec1, dtype=np.float64)
        vec2 = np.asarray(vec2, dtype=np.float64)
        
        # Check if vectors have the same dimension
        if vec1.shape != vec2.shape:
            raise ValueError("Vectors must have the same dimension")
        
        # Compute dot product and magnitudes
        dot_product = np.dot(vec1, vec2)
        magnitude1 = np.linalg.norm(vec1)
        magnitude2 = np.linalg.norm(vec2)
        
        # Check for zero magnitude to avoid division by zero
        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        
        # Compute and return cosine similarity
        return dot_product / (magnitude1 * magnitude2)
    
    except Exception as e:
        logging.error(f"Error in compute_cosine_similarity: {e}")
        return 0.0  # Return 0 similarity in case of any error

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
    
    prompt = f"""
    You are an AI assistant specializing in aviation. Provide detailed, thorough answers with examples
    where relevant. Use the context and history below to answer the user's question:
    
    Chat History and Full Context:
    {truncated_full_context}

    Context:
    {context}

    Human: {expanded_query}

    AI: Let me provide a detailed and informative answer:
    Format your response in the most appropriate structure:
    - Include relevant facts, explanations, and examples where appropriate. 
    - If it's about regulations, provide a **list of key FAA, ICAO, EASA, or MIL-STDs guidelines if available**.
    - If it's about an accident, provide a **summary of investigation insights**.
    - If it's about a technical issue, provide **a structured breakdown** with root causes.
    """
    # Calculate max tokens dynamically
    max_tokens = min(500, 4000 - len(truncated_full_context.split()))

    import random

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=max_tokens
            )
            response_text = response.choices[0].message.content.strip()

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
        
        response = "I'm sorry, but I couldn't generate a response due to an internal error."  # Default response

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

            # ✅ Fixed embedding retrieval logic inside the loop
            for batch in load_embeddings(EMBEDDINGS_FILE):  
                valid_embeddings = [emb for emb in batch if isinstance(emb, dict) and 'embedding' in emb and len(emb['embedding']) > 10]

                if valid_embeddings:
                    print(f"✅ Processing {len(valid_embeddings)} embeddings in batch...")
                    similarities = [compute_cosine_similarity(query_embedding, emb['embedding']) for emb in valid_embeddings]
                    top_results.extend(filter_and_rank_embeddings(valid_embeddings, similarities, top_n=10))  
                else:
                    logging.error("⚠️ No valid embeddings found! Skipping batch.")
                    continue  # Skip empty batches

                if not top_results:
                    logging.error(f"⚠️ No valid embeddings found for query: {expanded_query}")
                    print(f"⚠️ No relevant data found for: {expanded_query}")
                    response = "I'm sorry, but I couldn't generate a response due to missing data."
                    continue

                print("\nAviationAI:", response)  # ✅ Ensure response is displayed
                continue  # Skip this iteration instead of breaking the loop

            else:
                dynamic_top_n = get_dynamic_top_n(similarities)
                top_results = filter_and_rank_embeddings(embeddings, similarities, top_n=dynamic_top_n)

                unique_texts = set()
                combined_context = create_weighted_context(top_results)
                for result in top_results:
                    if result['text'] not in unique_texts:
                        unique_texts.add(result['text'])
                        combined_context += f"{result['text']}\n"

                # ✅ Include past chat history in the context
                chat_context = "\n".join([f"Human: {q}\nAI: {a}" for q, a in chat_history])

                # ✅ Combine everything into the final context
                full_context = f"{chat_context}\n\n{combined_context}"

                print("Generating response...")
                # ✅ Debugging: Print if response function is called
                logging.info("🛠️ Calling GPT-4 to generate response...")
                response = generate_response(combined_context, expanded_query, full_context, MODEL)

                if response and len(response) >= 10:
                    print("\nAviationAI:", response)  # ✅ Show response if valid
                    logging.info(f"✅ GPT-4 Response Generated: {response[:50]}...")  # ✅ Log first 50 characters for debug
                else:
                    logging.error("⚠️ GPT-4 returned an empty or invalid response!")
                    response = "I'm sorry, but I couldn't generate a response due to an internal error."
                    print("\nAviationAI:", response)

            is_helpful = get_user_feedback()
            if not is_helpful:
                print("I'm sorry the response wasn't helpful. Let me try to improve.")

            # ✅ Update chat history (keep only last 5 messages)
            chat_history.append((expanded_query, response))
            chat_history = chat_history[-max_history:]

        except Exception as e:
            logging.error(f"Error: {e}")
            print("\nAviationAI:", response)

        # ✅ Store chat history in AstraDB
        store_chat_in_db(session_id, expanded_query, response)
        print(f"Expanded Query: {expanded_query}")

if __name__ == "__main__":
    chat_loop()