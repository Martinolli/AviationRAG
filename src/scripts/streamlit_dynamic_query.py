import streamlit as st
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.probability import FreqDist
from nltk.text import Text
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up the OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load embeddings (modify as needed for your embeddings file path)
EMBEDDINGS_FILE = "data/embeddings/aviation_embeddings.json"

def load_embeddings():
    """Load embeddings from the JSON file."""
    with open(EMBEDDINGS_FILE, 'r') as f:
        return json.load(f)

def get_embedding(text, model="text-embedding-ada-002"):
    """Generate embedding using OpenAI's API."""
    try:
        response = client.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding: {e}")
        return None

def compute_similarity(embeddings, query_embedding):
    """Compute cosine similarity between query embedding and all document embeddings."""
    results = []
    for embedding in embeddings:
        similarity = cosine_similarity([query_embedding], [embedding['embedding']])[0][0]
        results.append({
            'chunk_id': embedding['chunk_id'],
            'filename': embedding['filename'],
            'text': embedding['text'],
            'similarity': similarity
        })
    return sorted(results, key=lambda x: x['similarity'], reverse=True)

# Streamlit UI setup
st.set_page_config(page_title="Aviation RAG Query Interface", layout="wide")
st.title("Aviation RAG Query Interface")

# Session state for query history
if 'query_history' not in st.session_state:
    st.session_state['query_history'] = []

# User query input
query = st.text_input("Enter your query:", placeholder="e.g., Describe latent failures in aviation accidents")

# Query submission
if st.button("Submit Query"):
    if query:
        st.session_state.query_history.append(query)

        # Generate query embedding
        st.write("Generating query embedding...")
        query_embedding = get_embedding(query)

        if query_embedding:
            # Load embeddings and compute similarities
            st.write("Loading embeddings...")
            embeddings = load_embeddings()
            
            st.write("Computing similarities...")
            try:
                results = compute_similarity(embeddings, query_embedding)

                # Display results
                st.subheader("Top Results")
                for result in results[:5]:
                    st.markdown(f"**Chunk ID**: {result['chunk_id']}")
                    st.markdown(f"**Filename**: {result['filename']}")
                    st.markdown(f"**Similarity**: {result['similarity']:.4f}")
                    st.markdown(f"**Text**: {result['text']}")
                    st.markdown("---")

            except Exception as e:
                st.error(f"Error computing similarities: {e}")
        else:
            st.error("Failed to generate embedding for the query.")

# Display query history
if st.session_state['query_history']:
    st.sidebar.title("Query History")
    for i, past_query in enumerate(st.session_state['query_history'], 1):
        st.sidebar.write(f"{i}. {past_query}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Version 1.0 | Streamlit Interface**")