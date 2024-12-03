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
