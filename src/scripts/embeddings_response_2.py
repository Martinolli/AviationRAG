import gevent.monkey
gevent.monkey.patch_all()
import json
import logging
import time
import random
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from nltk.corpus import wordnet
import uuid
from cassandra.cluster import Cluster
from cassandra.auth import PlainTextAuthProvider
from datetime import datetime

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Set up OpenAI API client
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY environment variable is not set.")
client = OpenAI(api_key=api_key)

# Connect to AstraDB
ASTRA_DB_SECURE_BUNDLE_PATH = os.getenv("ASTRA_DB_SECURE_BUNDLE_PATH")

def connect_to_astradb():
    """Establish a connection to AstraDB."""
    cloud_config = {'secure_connect_bundle': ASTRA_DB_SECURE_BUNDLE_PATH}
    cluster = Cluster(cloud=cloud_config)
    session = cluster.connect()
    session.set_keyspace("aviation_data")  # Use your keyspace name
    return session

# Initialize connection
session = connect_to_astradb()

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

def save_conversation(session_id, user_query, ai_response):
    """Save user-AI interaction to AstraDB."""
    timestamp = datetime.utcnow()
    query = """
    INSERT INTO aviation_data.aviation_conversation_history (session_id, timestamp, user_query, ai_response)
    VALUES (%s, %s, %s, %s);
    """
    session.execute(query, (session_id, timestamp, user_query, ai_response))
    print(f"Saved to DB: {user_query} → {ai_response}")

def retrieve_recent_conversations(session_id, limit=5):
    """Retrieve the most recent `limit` conversations for the given session."""
    query = f"""
    SELECT user_query, ai_response FROM aviation_data.aviation_conversation_history
    WHERE session_id = %s
    ORDER BY timestamp DESC
    LIMIT {limit};
    """
    rows = session.execute(query, (session_id,))
    return [{"user": row.user_query, "ai": row.ai_response} for row in rows]

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
        temperature=0.5,
        max_tokens=100
    ))

    return response.choices[0].message.content.strip() if response else query


def get_embedding(text):
    """Generate an embedding for the given text."""
    response = safe_openai_call(lambda: client.embeddings.create(
        input=[text],
        model="text-embedding-ada-002"
    ))
    return response.data[0].embedding if response else None

def load_embeddings(file_path):
    """Load embeddings from a JSON file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data if isinstance(data, list) and all(isinstance(item, dict) for item in data) else []
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Error loading embeddings: {e}")
        return []

def compute_cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    try:
        vec1, vec2 = np.array(vec1), np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    except Exception as e:
        logging.error(f"Error computing cosine similarity: {e}")
        return 0.0

def filter_and_rank_embeddings(embeddings, similarities, top_n=10, min_similarity=0.5):
    """
    Filter and rank embeddings based on similarity scores dynamically.
    """
    avg_similarity = np.mean(similarities)
    std_dev = np.std(similarities)  # Standard deviation for dynamic filtering
    dynamic_threshold = avg_similarity + (0.5 * std_dev)  # Adjust cutoff dynamically

    return sorted([
        {**emb, 'similarity': sim}
        for emb, sim in zip(embeddings, similarities)
        if sim > max(dynamic_threshold, min_similarity) and 'embedding' in emb
    ], key=lambda x: x['similarity'], reverse=True)[:top_n]


def create_weighted_context(top_results):
    """
    Create a weighted context for better response generation.
    
    Args:
        top_results (list): Retrieved documents sorted by similarity.

    Returns:
        str: Weighted context text.
    """
    combined_context = []
    total_weight = sum(result['similarity'] for result in top_results)
    
    for result in top_results:
        weight = result['similarity'] / total_weight
        weighted_text = f"{weight:.2f} - {result['text'][:500]}"  # Limit context to 500 chars per chunk
        combined_context.append(weighted_text)

    return "\n".join(combined_context)


def generate_structured_response(context, query, model="gpt-3.5-turbo"):
    """
    Generate a structured response using OpenAI.

    Args:
        context (str): Retrieved context.
        query (str): User query.
        model (str): GPT model version.

    Returns:
        str: Structured AI-generated response.
    """
    prompt = f"""
    You are an aviation expert AI. Provide a well-structured response based on the context provided.
    
    Context:
    {context}

    User Query:
    {query}

    Format your response in the most appropriate structure:
    - Provide information about Regulation Authorities guidelines if available.
    - When possible use aircraft accident examples to emphasize insights.
    - Provide a detailed explanation when necessary.
    - Establish organizational approach to explain issues in a structured manner.
    - If it's about a technical issue, provide a structured breakdown with root causes.

    Response:
    """
    response = safe_openai_call(lambda: client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=2000
    ))

    return response.choices[0].message.content.strip() if response else "I couldn't generate a response."

chat_memory = []

def update_chat_memory(user_input, ai_response, max_memory=5):
    """Store the last `max_memory` exchanges and log interactions."""
    if len(chat_memory) >= max_memory:
        chat_memory.pop(0)
    
    log_entry = f"User: {user_input}\nAI: {ai_response}\n"
    chat_memory.append(log_entry)

    with open("chat_history.log", "a", encoding="utf-8") as log_file:
        log_file.write(log_entry + "\n")


def track_conversation_context(user_input):
    """
    Detect follow-up questions for better context tracking, including requests for more details.
    """
    follow_up_keywords = [
        "compare", "difference", "related", "how does that", 
        "more details", "expand", "elaborate", "explain further"
    ]

    if any(keyword in user_input.lower() for keyword in follow_up_keywords):
        return "This appears to be a follow-up question. Here’s additional information:\n"
    
    return ""

def chat_loop():
    session_id = str(uuid.uuid4())  # Generate unique session ID
    EMBEDDINGS_FILE = "data/embeddings/aviation_embeddings.json"
    print("Welcome to AviationAI! Type 'exit' to quit.")
    
    embeddings = load_embeddings(EMBEDDINGS_FILE)
    if not embeddings:
        return

    # Retrieve past context from AstraDB
    past_exchanges = retrieve_recent_conversations(session_id)
    chat_history = [f"User: {ex['user']}\nAI: {ex['ai']}" for ex in past_exchanges]

    while True:
        query = input("\nUser: ")
        if query.lower() == 'exit':
            print("Goodbye!")
            break

        expanded_query = expand_query(query)
        query_embedding = get_embedding(expanded_query)
        if query_embedding is None:
            continue

        similarities = [compute_cosine_similarity(query_embedding, emb['embedding']) for emb in embeddings if 'embedding' in emb]
        top_results = filter_and_rank_embeddings(embeddings, similarities)
        context = create_weighted_context(top_results)

        # Append past context
        full_context = "\n".join(chat_history) + "\n" + context
        response = generate_structured_response(full_context, query)

        print("\nAviationAI:", response)

        # Save interaction to AstraDB
        save_conversation(session_id, query, response)

        # Update local chat history
        chat_history.append(f"User: {query}\nAI: {response}")

if __name__ == "__main__":
    chat_loop()

