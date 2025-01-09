import json
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Set up the OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text):
    """Generate embedding for the given text."""
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

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
    """Compute cosine similarity between two vectors."""
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

def filter_and_rank_embeddings(embeddings, similarities, top_n=10):
    """Filter and rank embeddings based on similarity scores."""
    return sorted(
        [
            {**emb, 'similarity': sim}
            for emb, sim in zip(embeddings, similarities)
            if sim > 0.5 and isinstance(emb, dict) and 'embedding' in emb  # Add type checking
        ],
        key=lambda x: x['similarity'],
        reverse=True
    )[:top_n]

def generate_response(context, query, model):
    """Generate a response using OpenAI."""
    prompt = f"""
    Context:
    {context}

    Question:
    {query}

    Provide a detailed response based on the context above.
    """
    try:
        if model in ["gpt-3.5-turbo", "gpt-4"]:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8
            )
            return response.choices[0].message.content.strip()
        elif model == "text-davinci-003":
            response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=0.7,
                max_tokens=500
            )
            return response.choices[0].text.strip()
        else:
            raise ValueError(f"Unsupported model: {model}")
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def chat_loop():
    EMBEDDINGS_FILE = "data/embeddings/aviation_embeddings.json"
    MODEL = "gpt-4"  # You can change this to "gpt-4" if available
    TOP_N = 10

    print("Welcome to the AviationAI Chat System!")
    print("Type 'exit' to end the conversation.")

    chat_history = []

    try:
        print("Loading embeddings...")
        embeddings = load_embeddings(EMBEDDINGS_FILE)
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return

    while True:
        QUERY_TEXT = input("\nUser: ")
        if QUERY_TEXT.lower() == 'exit':
            print("Thank you for using the Aviation RAG Chat System. Goodbye!")
            break

        try:
            print("Generating query embedding...")
            query_embedding = get_embedding(QUERY_TEXT)
            if query_embedding is None:
                raise ValueError("Failed to generate query embedding")

            print("Processing embeddings...")
            similarities = [compute_cosine_similarity(query_embedding, emb['embedding']) for emb in embeddings if isinstance(emb, dict) and 'embedding' in emb]
            top_results = filter_and_rank_embeddings(embeddings, similarities, top_n=TOP_N)

            unique_texts = set()
            combined_context = ""
            for result in top_results:
                if result['text'] not in unique_texts:
                    unique_texts.add(result['text'])
                    combined_context += f"{result['text']}\n"

            # Include chat history in the context
            chat_context = "\n".join([f"Human: {q}\nAI: {a}" for q, a in chat_history])
            full_context = f"{chat_context}\n\n{combined_context}"

            print("Generating response...")
            response = generate_response(full_context, QUERY_TEXT, MODEL)

            print("\nAviationAI:", response)

            # Update chat history
            chat_history.append((QUERY_TEXT, response))
            if len(chat_history) > 5:  # Keep only the last 5 exchanges
                chat_history = chat_history[-5:]

        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_loop()