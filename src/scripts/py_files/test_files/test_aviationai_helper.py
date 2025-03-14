import os
import logging
import json
import time
import openai
from dotenv import load_dotenv
import uuid
import subprocess
import sys
from AviationRAG.src.scripts.py_files.aviationai import client # Import client

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Embedding cache (now local to this file)
embedding_cache = {}

def get_embedding(text):
    """
    Gets the embedding for a given text, using the cache if available.
    """
    if text in embedding_cache:
        logging.info("✅ Using cached embedding for query.")
        return embedding_cache[text]
    else:
        logging.info("🔄 Generating new embedding for query...")
        response = client.embeddings.create(
            input=[text], # The input must be a list
            model="text-embedding-ada-002"
        )
        embedding = response.data[0].embedding
        embedding_cache[text] = embedding
        return embedding

def clear_cache():
    """
    Clears the embedding cache.
    """
    global embedding_cache
    embedding_cache = {}
    logging.info("🗑️ Embedding cache cleared.")
