import json
from openai import OpenAI
import numpy as np
from dotenv import load_dotenv
import os
import time
import random

# Load environment variables
load_dotenv()

# Set up the OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def expand_query(query):
    prompt = f"Expand the following query with relevant aviation terms: {query}"
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=750
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error expanding query: {e}")
        return query

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
        print(f"Error generating embedding: {e}")
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
        print(f"Error in compute_cosine_similarity: {e}")
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
    max_context_length = 4000  # Adjust this value based on your needs
    max_retries = 3
    base_delay = 1

    # Implement a sliding window for chat history
    chat_history = full_context.split("\n\n")[-5:]  # Keep last 5 exchanges
    truncated_full_context = "\n\n".join(chat_history)


    # Truncate the context if it's too long
    if len(truncated_full_context) > max_context_length:
        truncated_full_context = truncated_full_context[-max_context_length:]
    
    prompt = f"""
    You are an AI assistant specializing in aviation. Provide detailed, thorough answers with examples where relevant. Use the context and history below to answer the user's question:

    Chat History and Full Context:
    {truncated_full_context}

    Context:
    {context}

    Human: {query}
    AI: Let me provide a detailed and informative answer:
    """

    for attempt in range(max_retries):
        try:
            if model in ["gpt-3.5-turbo", "gpt-4"]:
                response = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.7,
                    max_tokens=2000  # Reduced from 150 to 100
                )
                return response.choices[0].message.content.strip()
            else:
                raise ValueError(f"Unsupported model: {model}")
        except Exception as e:
            if attempt < max_retries - 1:
                delay = (base_delay * 2 ** attempt) + (random.randint(0, 1000) / 1000.0)
                print(f"Error generating response: {e}. Retrying in {delay:.2f} seconds...")
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

    chat_history = []
    max_history = 5

    try:
        print("Loading embeddings...")
        embeddings = load_embeddings(EMBEDDINGS_FILE)
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return

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

            # Include chat history in the context
            chat_context = "\n".join([f"Human: {q}\nAI: {a}" for q, a in chat_history])
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
            print(f"Error: {e}")

if __name__ == "__main__":
    chat_loop()