from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Debug: Check if the API key is loaded
api_key = os.getenv("OPENAI_API_KEY")
if api_key:
    print("API key loaded successfully.")
else:
    print("Failed to load API key.")

from openai import OpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Generate embedding
try:
    response = client.embeddings.create(
        input=["aircraft safety management system"],
        model="text-embedding-ada-002"
    )
    print("Response object type:", type(response))
    print("Response contents:", response)
except Exception as e:
    print(f"Error: {e}")
