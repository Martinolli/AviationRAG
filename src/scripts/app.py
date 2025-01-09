from flask import Flask, request, jsonify
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
import logging
from flask_cors import CORS

# Load environment variables
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)
CORS(app)

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s'
)
app.logger.setLevel(logging.INFO)


# Function to generate embedding for a given text
def get_embedding(text, model="text-embedding-ada-002"):
    try:
        response = client.embeddings.create(input=[text], model=model)
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Function to load embeddings from a JSON file
def load_embeddings(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Embeddings file not found: {file_path}")
    with open(file_path, 'r', encoding='utf-8') as file:
        return json.load(file)

# Function to compute cosine similarity
def compute_cosine_similarity(query_embedding, embeddings):
    query_vector = np.array(query_embedding).reshape(1, -1)
    embedding_vectors = np.array([item['embedding'] for item in embeddings])
    return cosine_similarity(query_vector, embedding_vectors)[0]

def generate_response(context, query, model="gpt-3.5-turbo"):
    """
    Generate a response using OpenAI based on the retrieved context.
    """
    max_context_length = 4000  # Adjust to fit the model's input limit
    truncated_context = context[:max_context_length]

    prompt = f"""
    Context:
    {truncated_context}

    Question:
    {query}

    Provide a detailed, accurate, and concise response based on the context above. Include examples or explanations as necessary.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        app.logger.error(f"Error generating response: {str(e)}")
        return "Failed to generate response."


# API endpoint to handle queries
# Add a root route for testing
@app.route('/')
def home():
    return "Welcome to the Aviation RAG API"

# Modify the query route to include both GET and POST methods
@app.route('/query', methods=['GET', 'POST'])
def query():
    if request.method == 'GET':
        app.logger.info("GET request received at /query")
        return "This is the query endpoint. Please use POST method to submit a query."
    
    try:
        data = request.json
        if not data or 'query' not in data:
            app.logger.warning("No query provided in the request")
            return jsonify({"error": "No query provided"}), 400
        
        query_text = data['query']
        model = data.get('model', 'text-embedding-ada-002')
        page = int(data.get('page', 1))  # Default to page 1
        page_size = int(data.get('page_size', 10))  # Default to 10 results per page
        embeddings_file = "data/embeddings/aviation_embeddings.json"

        # Generate embedding for the query
        query_embedding = get_embedding(query_text, model)
        if not query_embedding:
            return jsonify({"error": "Failed to generate query embedding"}), 500

        # Load embeddings
        embeddings = load_embeddings(embeddings_file)

        # Compute similarity
        similarities = compute_cosine_similarity(query_embedding, embeddings)
        results = [
            {
                "chunk_id": emb["chunk_id"],
                "filename": emb["filename"],
                "text": emb["text"],
                "similarity": float(sim)
            }
            for emb, sim in zip(embeddings, similarities)
        ]
        results = sorted(results, key=lambda x: x["similarity"], reverse=True)

        # Apply pagination
        start = (page - 1) * page_size
        end = start + page_size
        paginated_results = results[start:end]

        # Combine context for response generation
        combined_context = "\n".join([result["text"] for result in paginated_results])

        # Generate the final response using OpenAI
        generated_response = generate_response(combined_context, query_text, model)

        app.logger.info(f"Returning {len(paginated_results)} results for query: {query_text}")
        return jsonify({
            "page": page,
            "page_size": page_size,
            "total_results": len(results),
            "results": paginated_results,
            "response": generated_response
        })

    except Exception as e:
        app.logger.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')  # Listens on all interfaces
