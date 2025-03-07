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
import faiss

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
    """Generate a structured compliance-driven response using GPT-4."""

    prompt = f"""
    🛠️ **Aviation Compliance Expert Analysis**
    
    You are an AI specializing in **aviation safety, compliance, and regulatory risk analysis**. 
    When answering user queries, follow this structured approach:
    
    ✈️ **1️⃣ Issue Analysis**  
    - Identify the key aviation compliance concern in the query.  
    - Analyze potential procedural gaps, risk factors, or human error aspects.
    - Identify the core **aviation safety concerns** in the query.  
    - Assess potential **regulatory violations or procedural gaps** 

    📑 **2️⃣ Regulatory Review (FAA, ICAO, EASA)**  
    - Determine which aviation regulations, SARPs (Standards & Recommended Practices), or compliance frameworks apply.  
    - Compare the issue against **ICAO Annex 19, FAA Part 5, EASA safety frameworks**.  
    - Identify compliance gaps and industry best practices.  

    🔍 **3️⃣ Cross-Check with Accident Reports**  
    - Retrieve relevant aviation accident reports (if applicable) to compare with the current issue.  
    - Identify lessons learned from past cases.  

    🛠️ **4️⃣ Risk Mitigation Framework & Safety Enhancements**  
    - **Communication Enhancement:** Require ATC clearances to be confirmed using ICAO Standard Phraseology.  
    - **Pilot Decision-Making Protocols:** Ensure that all captains follow Crew Resource Management (CRM) best practices.  
    - **Pre-Flight Risk Assessments:** Mandate runway safety briefings during high-traffic periods to avoid misunderstandings.  
    - **Safety Culture Improvements:** Conduct SMS workshops to align operational behavior with FAA Part 5 and ICAO Annex 19.  
    
    🛠️ **5️⃣ Compliance Validation Score & Risk Level**  
    The compliance validation score is determined based on:  
    - ✅ **Adherence to ICAO, FAA, and EASA regulations**  
    - ✅ **Severity of human error or procedural gaps**  
    - ✅ **Historical trends in similar accidents**  

    📊 **Final Risk & Compliance Score:**  
    - **1-3 (High Risk, Non-Compliant)** – Significant violations and poor safety culture.  
    - **4-6 (Moderate Risk, Partially Compliant)** – Some procedural weaknesses that need improvement.  
    - **7-10 (Low Risk, Compliant)** – Strong safety culture with minor procedural gaps.  

    **This scenario has a compliance validation score of 6/10, indicating moderate SMS failures.**  

    🔍 **6️⃣ Cross-Check with Accident Investigations**  
    - Use retrieved reports to compare **similar risk factors, human errors, and procedural failures**.
    - Cross-reference at least **two similar aviation accidents** and highlight common safety failures.    

    Here is the user's question:
    {query}

    Ensure your response is well-structured, factual, and backed by aviation regulations.
    """
    
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,  # ✅ Lower temperature for more factual answers
            max_tokens=2500  # ✅ Allow longer responses if needed
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
        print(" - 'Analyze how ICAO Annex 19 applies to accident prevention.'")
        print(" - 'What are the SMS compliance lessons from the Tenerife disaster?'")
        print(" - 'Compare FAA Part 43 maintenance regulations with EASA Part M.'")
        print(" - 'Assess if this risk management process aligns with FAA Part 5.'")


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
        # ✅ Normalize query embedding before search
        normalized_query_embedding = faiss_index.normalize_vector(query_embedding)

        # ✅ Retrieve more potential matches, then rank them
        indices, distances = faiss_index.index.search(normalized_query_embedding, k=20)

        # ✅ Post-process: Filter out weak matches based on similarity score
        valid_results = [(idx, 1 / (1 + dist)) for idx, dist in zip(indices.flatten(), distances.flatten()) if dist < 0.4]

        # ✅ Sort by similarity score (higher = more relevant)
        valid_results.sort(key=lambda x: x[1], reverse=True)

        # ✅ Return top 5 most relevant results
        results = valid_results[:7]

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
