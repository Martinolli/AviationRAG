from flask import Flask, request, jsonify
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Set up the OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.route('/')
def index():
    """
    Home route to welcome users and provide usage instructions.
    """
    return jsonify({
        "message": "Welcome to the Aviation Backend API!",
        "usage": {
            "POST /query": "Send a JSON payload with 'query' to process embeddings."
        }
    })

@app.route('/query', methods=['POST'])
def query():
    """
    Endpoint to process user queries using the OpenAI API.
    """
    try:
        # Parse JSON payload
        data = request.json
        query = data.get('query', '')

        # Validate the query
        if not query:
            return jsonify({"error": "Please provide a valid query."}), 400

        # Call OpenAI API to process the query
        response = client.chat.completions.create(
            model="gpt-4",  # Use "gpt-3.5-turbo" if GPT-4 is not available
            messages=[
                {"role": "system", "content": "You are an aviation expert AI assistant."},
                {"role": "user", "content": query}
            ],
            max_tokens=150
        )

        # Extract and return the result
        result = response.choices[0].message.content
        return jsonify({"query": query, "response": result})
    except Exception as e:
        # Handle and log any errors
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)