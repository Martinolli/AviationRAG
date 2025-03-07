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
console_handler.setLevel(logging.INFO)

for handler in [info_log, error_log, performance_log, console_handler]:
    handler.setFormatter(log_formatter)

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
logger.addHandler(info_log)
logger.addHandler(error_log)
logger.addHandler(performance_log)
logger.addHandler(console_handler)

# ✅ Initialize OpenAI API Client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ✅ Path to embeddings file
EMBEDDINGS_FILE = "data/embeddings/aviation_embeddings.json"

# ✅ Load and create FAISS index
try:
    faiss_index = FAISSIndexer.load_from_file(EMBEDDINGS_FILE)
    logging.info(f"✅ FAISS index created with {faiss_index.index.ntotal} embeddings.")
except Exception as e:
    logging.error(f"❌ Error creating FAISS index: {e}")
    exit(1)

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

        print("Chat stored successfully in AstraDB!")
        logging.info(f"💾 Storing chat for session: {session_id} | Query: {user_query[:50]}...")

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
        print("⚠️ Warning: `session_id` is empty! Generating a new one...")
        session_id = str(uuid.uuid4())  # Assign a new session if empty

    chat_data = {
        "action": "retrieve",
        "session_id": session_id,
        "limit": limit
    }

    try:
        # Use UTF-8 encoding when running the command
        result = subprocess.run(['python', 'src/scripts/py_files/retrieve_chat.py', session_id], 
                                capture_output=True, text=True, encoding='utf-8')
        
        # Check if stdout is None before calling strip()
        if result.stdout is not None:
            output = result.stdout.strip()
            if output:
                return eval(output)
        
        # If we reach here, either stdout was None or empty
        print(f"No chat history found for session {session_id}")
        return []
    
    except subprocess.CalledProcessError as e:
        print(f"Error retrieving chat history: {e}")
        return []
    except Exception as e:
        print(f"Unexpected error: {e}")
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
def generate_response(query, model="gpt-4"):
    """Generate a response using GPT-4 with compliance-focused analysis."""

    prompt = f"""
    You are an aviation compliance expert specializing in safety regulations,
    accident analysis, and procedural risk assessments.
    
    When answering the user's query, you must, when applicable:
    - Analyze the issue from an aviation compliance perspective.
    - Compare the problem with known accident cases stored in FAISS.
    - Identify likely regulatory gaps using FAA, ICAO, EASA standards,
    or the Standards and Recomended Practices in Aviation Industy.
    - Provide structured recommendations based on safety best practices.

    Here is the question you must analyze:
    {query}

    🚀 **Ensure your answer is structured as follows:**
    1️⃣ **Analysis of the Issue**
    2️⃣ **Comparison with Similar Accidents**
    3️⃣ **Regulatory Compliance Review**
    4️⃣ **Recommendations for Safety Improvement**
    Remarks: If the request is too broad, focus on the most critical aspects.
    The answer could be informative, according to the question asked.    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5,
            max_tokens=2000
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
    
    while True:
        print("\n🔹 Suggested Topics: ")
        print(" - 'Analyze an accident similar to Helios Airways Flight 522.'")
        print(" - 'What regulatory gaps led to the Tenerife Airport Disaster?'")
        print(" - 'Does this maintenance procedure comply with FAA Part 43?'")
        print(" - 'How does SMS align with ICAO Annex 19?'")

        query = input("\nUser: ")

        if query.lower() == 'exit':
            print("Goodbye!")
            break

        logging.info("Generating query embedding...")
        query_embedding = get_embedding(query)

        if query_embedding is None:
            print("⚠️ Embedding generation failed. Try again.")
            continue

        logging.info("Searching FAISS for relevant documents...")
        results = faiss_index.search(query_embedding, k=5)

        # ✅ Format retrieved results for GPT-4
        context_texts = "\n".join([f"- {res['text']}" for res, _ in results])

        formatted_query = f"""
        Based on the following aviation documents:

        {context_texts}

        Question:
        {query}
        """

        print("Generating response...")
        response = generate_response(formatted_query, "gpt-4")
        print("\nAviationAI:", response)

        store_chat_in_db(session_id, query, response)

# ✅ Run the chat loop
if __name__ == "__main__":
    chat_loop()
