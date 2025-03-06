import logging
from faiss_indexer import FAISSIndexer
import os
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up the OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# Path to your embeddings file
EMBEDDINGS_FILE = "data/embeddings/aviation_embeddings.json"

# Load and create FAISS index
try:
    faiss_index = FAISSIndexer.load_from_file(EMBEDDINGS_FILE)
    logging.info(f"✅ FAISS index created with {faiss_index.index.ntotal} embeddings.")
except Exception as e:
    logging.error(f"❌ Error creating FAISS index: {e}")
    exit(1)

# Test FAISS with a sample query
sample_query_embedding = faiss_index.index.reconstruct(0)  # Get the first embedding as a sample query
results = faiss_index.search(sample_query_embedding, k=5)

logging.info("🔍 Testing FAISS retrieval with a sample query...")
for result, distance in results:
    logging.info(f"Chunk ID: {result['chunk_id']}, Distance: {distance}")

# Your model interaction function here

def get_embedding(text, model="text-embedding-ada-002"):
    """
    Generate embedding using OpenAI's updated client.

    Args:
        text (str): The input text to generate an embedding for.
        model (str): The embedding model to use.

    Returns:
        list: The generated embedding.
    """
    try:
        response = client.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None
    
def generate_response(query, model = "gpt-4"):
    """
    Generate a response using OpenAI.

    Args:
        context (str): The context string generated from retrieved chunks.
        query (str): The user query.
        model (str): The model to use for generating a response.

    Returns:
        str: The generated response from OpenAI.
    """
    # max_context_length = 8000  # Adjust based on the model's capabilities
    # truncated_context = context[:max_context_length]
    
    prompt = f"""
    
    Question:
    {query}

    Provide a detailed, comprehensive, and accurate response based on the context above. 
    Include relevant facts, explanations, and examples where appropriate. 
    For each key piece of information in your response, cite the source document in square brackets, 
    e.g., [Document: Safety Manual]. If information comes from multiple sources, list all relevant sources.
    If the context doesn't contain enough information to fully answer the question, 
    clearly state what information is missing or uncertain.
    """
    try:
        if model in ["gpt-3.5-turbo", "gpt-4"]:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        elif model == "text-davinci-003":
            response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=0.7,
                max_tokens=1000
            )
            return response.choices[0].text.strip()
        else:
            raise ValueError(f"Unsupported model: {model}")
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def interact_with_model(query):
    # Generate embedding for the query
    query_embedding = get_embedding(query)  # You'll need to implement this function
    
    # Search similar embeddings
    results = faiss_index.search(query_embedding, k=5)
    
    # Use the results to inform your model's response
    print("Generating response...")
    response = generate_response(results, "gpt-4")  # You'll need to implement this function
    print("\nGenerated Response:")
    print(response)


# Main application loop
if __name__ == "__main__":
    while True:
        query = input("Enter your query (or 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        response = interact_with_model(query)
        print(response)