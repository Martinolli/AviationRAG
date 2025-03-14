import logging
import os
import numpy as np
import json
import time
import uuid
import sys
import subprocess
from openai import OpenAI, OpenAIError
from dotenv import load_dotenv
from faiss_indexer import FAISSIndexer
import tiktoken

# ✅ Load environment variables
load_dotenv()

# ✅ Define base paths
base_dir = r'C:\Users\Aspire5 15 i7 4G2050\ProjectRAG\AviationRAG'
log_dir = os.path.join(base_dir, 'logs')
chat_dir = os.path.join(base_dir, 'chat')
chat_id = os.path.join(base_dir, 'chat_id')

# ✅ Ensure directories exist
for directory in [log_dir, chat_dir, chat_id]:
    if not os.path.exists(directory):
        os.makedirs(directory)

# ✅ Configure logging
info_log_path = os.path.join(log_dir, 'info.log')
error_log_path = os.path.join(log_dir, 'error.log')
performance_log_path = os.path.join(log_dir, 'performance.log')

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
logger.setLevel(logging.DEBUG)
logger.addHandler(info_log)
logger.addHandler(error_log)
logger.addHandler(performance_log)
logger.addHandler(console_handler)

import logging

# Suppress verbose logging from OpenAI, urllib3, and httpx (used internally)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)


# ✅ Initialize OpenAI API Client
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY")) # The client is now in this file

# ✅ Path to embeddings file
EMBEDDINGS_FILE = "data/embeddings/aviation_embeddings.json"

# ✅ Load and create FAISS index
try:
    faiss_index = FAISSIndexer.load_from_file(EMBEDDINGS_FILE, verbose=False)
    logging.info(f"✅ FAISS index created with {faiss_index.index.ntotal} embeddings.")
except Exception as e:
    logging.error(f"❌ Error creating FAISS index: {e}")
    exit(1)

def num_tokens_from_string(string: str, encoding_name: str = "cl100k_base") -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

def parse_json_output(output):
    first_brace = output.find("{")
    if first_brace != -1:
        output = output[first_brace:]
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
    
# ✅ Function to store chat history in AstraDB
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

        # logging.info(f"💾 Storing chat for session: {session_id} | Query: {user_query[:50]}...")

    except subprocess.CalledProcessError as e:
        logging.error(f"Error storing chat: {e}")

# ✅ Function to retrieve chat history from AstraDB
def retrieve_chat_from_db(session_id, limit=5):
    """
    Calls the Node.js script to retrieve chat history from AstraDB.
    """
    script_path = os.path.join(os.path.dirname(__file__), '..', 'js_files', 'store_chat.js')
    logging.info(f"📥 Retrieving chat messages for session: {session_id}")

    if not session_id.strip():
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
        return parse_json_output(result.stdout.strip())

    except subprocess.CalledProcessError as e:
        logging.error(f"Error retrieving chat: {e}")
        return []

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
def generate_response(query, context, model="gpt-4-turbo"):
    """Generate a structured compliance-driven response using GPT-4."""
    
    prompt = f"""
    🛠️ **Aviation Compliance Expert Analysis**
    
    You are an AI specializing in aviation safety, compliance, and regulatory risk analysis. 
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

    Always prioritize accuracy and relevance in your responses. Use the provided context 
    to support your answers, but feel free to draw on your general knowledge of aviation 
    safety and regulations when appropriate.
    
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
            temperature=0.6,  # ✅ Lower temperature for more factual answers
            max_tokens=2000  # ✅ Allow longer responses if needed
        )
        return response.choices[0].message.content.strip()
    except OpenAIError as e:
        logging.error(f"❌ Error calling GPT-4: {e}")
        return "I'm sorry, but I encountered an error while generating a response."


# ✅ Function to handle chat loop
def chat_loop():
    """Interactive chat loop for AviationAI."""
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
        query = input("\nUser: ").strip()

        if query.lower() == 'exit':
            print("Thank you for using the Aviation RAG Chat System. Goodbye!")
            break

        try:
            logging.info("Generating query embedding...")
            query_embedding = get_embedding(query)
            if query_embedding is None:
                raise ValueError("Failed to generate query embedding")

            logging.info("Searching FAISS for relevant documents...")
            results = faiss_index.search(query_embedding, k=15)
            context_texts = []
            total_tokens = 0
            max_tokens = 6000  # Leave room for the query and response
            for metadata, score in results:
                doc_text = metadata['text']
                doc_tokens = num_tokens_from_string(doc_text)
                if total_tokens + doc_tokens > max_tokens:
                    break
                context_texts.append(doc_text)
                total_tokens += doc_tokens
                
            context = "\n".join(context_texts)
      
           # Generate response
            response = generate_response(query, context)
            print("\nAviationAI:", response)
            # Store chat in AstraDB
            store_chat_in_db(session_id, query, response)
            # Update chat history
            past_exchanges.append((query, response))
            if len(past_exchanges) > 5:
                past_exchanges = past_exchanges[-5:]

        except Exception as e:
            logging.error(f"Error: {e}")
            print(f"An error occurred: {e}")
        chat_history.append((query, response))
        chat_history = chat_history[-max_history:]
# ✅ Run the chat loop
if __name__ == "__main__":
    chat_loop()
