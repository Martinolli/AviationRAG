import json
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import os
import time
import random
from nltk.corpus import wordnet
import logging
import datetime
import subprocess
import uuid

# Load environment variables
load_dotenv()

# Define absolute paths
base_dir = r'C:\Users\Aspire5 15 i7 4G2050\ProjectRAG\AviationRAG'
log_dir = os.path.join(base_dir, 'logs')  # Define the path to the logs folder
chat_dir = os.path.join(base_dir, 'chat')  # Define the path to the chat folder
chat_id = os.path.join(base_dir, 'chat_id')  # Define the path to the chat folder

# Ensure the chat directory exists
if not os.path.exists(chat_dir):
    os.makedirs(chat_dir)

if not os.path.exists(chat_id):
    os.makedirs(chat_id)

# Set up logging
log_file_path = os.path.join(log_dir, 'chat_system.log')
logging.basicConfig(filename=log_file_path, level=logging.INFO,
                   format='%(asctime)s - %(levelname)s - %(message)s')


# Set up the OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def store_chat_in_db(session_id, user_query, ai_response):
    """
    Calls the Node.js script to store chat in AstraDB.
    """
    # Define the correct path to store_chat.js inside src/scripts/
    
    script_path = os.path.join(os.path.dirname(__file__), "store_chat.js")

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
            check=True
        )
        print("Chat stored successfully in AstraDB!")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error storing chat: {e}")

def retrieve_chat_from_db(session_id, limit=5):
    """
    Calls the Node.js script to retrieve chat history from AstraDB.
    """
    script_path = os.path.join(os.path.dirname(__file__), "store_chat.js")

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

        try:
            parsed_output = json.loads(output)
            if parsed_output.get("success", False):
                messages = parsed_output.get("messager", [])

                # If multiple messages exist, merge them into a single text block
                if len(messages) > 1:
                    combined_text = "\n\n".join(
                        f"User: {msg['user_query']}\nAI: {msg['ai_response']}"
                        for msg in messages
                    )
                    return[{"combined_test": combined_text}]
                return messages
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

def store_chat_history(chat_history):
    current_date = datetime.datetime.now().strftime("%Y-%m-%d")
    log_filename = os.path.join(chat_dir, f"chat_history_{current_date}.log")
    
    with open(log_filename, "a") as file:
        for query, response in chat_history:
            timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            file.write(f"[{timestamp}] Human: {query}\n")
            file.write(f"[{timestamp}] AI: {response}\n\n")


def generate_chat_summary(chat_history):
    # Extract the most common queries and responses from the chat history
    # You can use a library like NLTK or spaCy to perform this task
    # For simplicity, let's assume we're using a simple dictionary to store the queries and responses
    summary_dict = {}

    for query, response in chat_history:
        if query in summary_dict:
            summary_dict[query].append(response)
        else:
            summary_dict[query] = [response]

    # Generate a summary of the conversation based on the summary_dict
    summary = ""
    for query, responses in summary_dict.items():
        summary += f"{query}: {', '.join(responses)}\n"

    return summary

def expand_query(query):
    """
    Expand the user query using aviation-specific terminology and synonyms.
    """
    synonyms = set()
    for word in query.split():
        for syn in wordnet.synsets(word):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().replace('_', ' '))  # Add synonyms

    prompt = f"""
    Given the aviation-related query below, expand it using technical aviation terms and related synonyms.

    Query: {query}
    Expanded terms: {', '.join(synonyms)}
    
    Final Expanded Query:
    """
    response = safe_openai_call(lambda: client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=500
    ))

    return response.choices[0].message.content.strip() if response else query

def get_user_feedback():
    while True:
        feedback = input("Was this response helpful? (y/n): ").lower()
        if feedback in ['y', 'n']:
            return feedback == 'y'
        print("Please enter 'y' for yes or 'n' for no.")

def get_embedding(text):
    """Generate embedding for the given text."""
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
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


def load_embeddings(file_path):
    """Load embeddings from a JSON file."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Ensure the loaded data is a list of dictionaries
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return data
    else:
        raise ValueError("Loaded data is not in the expected format (list of dictionaries)")

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

def filter_and_rank_embeddings(embeddings, similarities, top_n=10, min_similarity=0.5):
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

def generate_response(context, query, full_context, model):
    """Generate a response using OpenAI."""
    max_context_length = 8000  # Adjust this value based on your needs
    max_retries = 3
    base_delay = 1

    # Implement a sliding window for chat history
    chat_history = full_context.split("\n\n")[-5:]  # Keep last 5 exchanges
    truncated_full_context = "\n\n".join(chat_history)


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

    Human: {query}

    AI: Let me provide a detailed and informative answer:
    Format your response in the most appropriate structure:
    - If it's about regulations, provide a **list of key FAA or ICAO guidelines if available**.
    - If it's about an accident, provide a **summary of investigation insights**.
    - If it's about a technical issue, provide **a structured breakdown** with root causes.
    """
    # Calculate max tokens dynamically
    max_tokens = min(1000, 4000 - len(truncated_full_context.split()))

    for attempt in range(max_retries):
        try:
            if model in ["gpt-3.5-turbo", "gpt-4"]:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=max_tokens  # Reduced from 150 to 100
                )
                return response.choices[0].message.content.strip()
            else:
                raise ValueError(f"Unsupported model: {model}")
        except Exception as e:
            if attempt < max_retries - 1:
                delay = (base_delay * 2 ** attempt) + (random.randint(0, 1000) / 1000.0)
                logging.error(f"Error generating response: {e}. Retrying in {delay:.2f} seconds...")
                time.sleep(delay)
            else:
                print(f"Error generating response after {max_retries} attempts: {e}")
                return "I apologize, but I'm having trouble generating a response at the moment. Please try again later."

    return None

def chat_loop():
    EMBEDDINGS_FILE = "data/embeddings/aviation_embeddings.json"
    MODEL = "gpt-3.5-turbo"  # You can change this to "gpt-4" if available
    
    print("Welcome to the AviationAI Chat System!")
    print("Type 'exit' to end the conversation.")

    # Check if there’s an active session (previous exchanges exist)
    existing_session_file = os.path.join(chat_id, "last_session_id.txt")

    if os.path.exists(existing_session_file):
        with open(existing_session_file, "r") as file:
            session_ids = file.read()
        if session_ids:
            last_session_id = session_ids[-1].strip()
            print("Found an existing session.")
            continue_previous = input("Do you want to continue the previous chat? (y/n): ").lower() == 'y'
            if continue_previous:
                session_id = last_session_id
                past_exchanges = retrieve_chat_from_db(session_id)
            else:
                session_id = str(uuid.uuid4())
                past_exchanges = []
        else:
            session_id = str(uuid.uuid4())
            past_exchanges = []
    else:
        session_id = str(uuid.uuid4())
        past_exchanges = []
    
    # Save the current session_id for future use
    with open(existing_session_file, "a") as file:
        file.write(f"{session_id}\n")
    
    try:
        print("Loading embeddings...")
        embeddings = load_embeddings(EMBEDDINGS_FILE)
    except Exception as e:
        logging.error(f"Error loading embeddings: {e}")
        return
    
    # Retrieve past conversation history for the session
    chat_history = [f"User: {ex['user_query']}\nAI: {ex['ai_response']}" for ex in past_exchanges]
    max_history = 5  # Maximum number of chat exchanges to keep in history

    while True:
        QUERY_TEXT = input("\nUser: ")
        if QUERY_TEXT.lower() == 'exit':
            print("Thank you for using the AviationAI Chat System. Goodbye!")
            break

        try:
            print("Generating query embedding...")
            expanded_query = expand_query(QUERY_TEXT)
            query_embedding = get_embedding(QUERY_TEXT)
            if query_embedding is None:
                raise ValueError("Failed to generate query embedding")

            print("Processing embeddings...")
            similarities = [compute_cosine_similarity(query_embedding, emb['embedding']) for emb in embeddings if isinstance(emb, dict) and 'embedding' in emb]
            dynamic_top_n = get_dynamic_top_n(similarities)
            top_results = filter_and_rank_embeddings(embeddings, similarities, top_n=dynamic_top_n)

            unique_texts = set()
            combined_context = create_weighted_context(top_results)
            for result in top_results:
                if result['text'] not in unique_texts:
                    unique_texts.add(result['text'])
                    combined_context += f"{result['text']}\n"

            # Include past chat history in the context
            chat_context = "\n".join([f"Human: {q}\nAI: {a}" for q, a in chat_history])

            # Combine everything into the final context
            full_context = f"{chat_context}\n\n{combined_context}"


            print("Generating response...")
            response = generate_response(combined_context, QUERY_TEXT, full_context, MODEL)

            print("\nAviationAI:", response)

            is_helpful = get_user_feedback()
            if not is_helpful:
                print("I'm sorry the response wasn't helpful. Let me try to improve.")
            # Here you could implement logic to refine the response or adjust parameters

            # Update chat history
            chat_history.append((QUERY_TEXT, response))
            if len(chat_history) > max_history:  # Keep only the last 5 exchanges
                chat_history = chat_history[-max_history:]

        except Exception as e:
            logging.error(f"Error: {e}")

        # Generate a summary of the conversation
        summary = generate_chat_summary(chat_history)

        # Store chat history in a log file
        chat_history.append((QUERY_TEXT, response))
        if len(chat_history) > max_history:  # Keep only the last 5 exchanges
            chat_history = chat_history[-max_history:]

        store_chat_history(chat_history)
        store_chat_in_db(session_id, QUERY_TEXT, response)
        
if __name__ == "__main__":
    chat_loop()