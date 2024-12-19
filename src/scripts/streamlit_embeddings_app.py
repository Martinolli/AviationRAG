import streamlit as st
import json
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.probability import FreqDist
from nltk.text import Text
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from openai import OpenAI
from dotenv import load_dotenv
import pickle

# Load environment variables
load_dotenv()

# Set up the OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-ada-002"):
    """Generate embedding using OpenAI's updated client."""
    try:
        response = client.embeddings.create(
            input=[text],  # OpenAI API requires input as a single string or list
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Load embeddings
def load_embeddings(embeddings_file):
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        embeddings_data = json.load(f)
    return embeddings_data

# Calculate similarities
def similarity_search(embeddings, query_embedding, top_n=5):
    embeddings_array = np.array([np.array(chunk["embedding"]) for chunk in embeddings])
    similarities = cosine_similarity([query_embedding], embeddings_array).flatten()
    top_indices = similarities.argsort()[-top_n:][::-1]
    return [(embeddings[i], similarities[i]) for i in top_indices]

# Generate frequency distribution plot
def plot_frequency_distribution(tokens):
    fdist = FreqDist(tokens)
    top_words = fdist.most_common(10)

    words, counts = zip(*top_words)
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts)
    plt.title("Top 10 Words by Frequency")
    plt.xlabel("Words")
    plt.ylabel("Frequency")
    st.pyplot(plt)

# Generate concordance
def display_concordance(corpus_tokens, word, window=50):
    """Display concordance for a word from embeddings data."""
    st.markdown(f"### Concordance for '{word}'")
    text_obj = nltk.Text(corpus_tokens)
    concordance_results = text_obj.concordance_list(word, width=window)
    
    if concordance_results:
        for result in concordance_results:
            st.write("... " + result.line.strip() + " ...")
    else:
        st.write(f"No concordance found for '{word}'.")


# Main Streamlit application
def main():
    st.title("AviationRAG: Embedding and Textual Analysis Tool")

    # File paths
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    EMBEDDINGS_FILE = os.path.join(BASE_DIR, '..', '..', 'data', 'embeddings', 'aviation_embeddings.json')
    PKL_FILENAME = os.path.join(BASE_DIR, '..', '..', 'data', 'raw', 'aviation_corpus.pkl')

    # Load data
    st.sidebar.header("Data Loading")
    embeddings = load_embeddings(EMBEDDINGS_FILE)
    st.sidebar.success("Embeddings Loaded")

    with open(PKL_FILENAME, 'rb') as file:
        corpus_data = pickle.load(file)
        corpus_tokens = [token for doc in corpus_data for token in doc.get("tokens", [])]
    st.sidebar.success("Corpus Loaded")

    # Query input
    query = st.text_input("Enter your query text:")

    if query:
        # Generate query embedding
        st.info("Generating query embedding...")
        query_embedding = get_embedding(query)
        if query_embedding is None:
            st.error("Failed to generate query embedding.")
            return
        
        # Embedding similarity search
        st.subheader("Embedding Similarity Results")
        try:
            results = similarity_search(embeddings, query_embedding)
            for result, similarity in results:
                st.write(f"Chunk ID: {result['chunk_id']}\nFilename: {result['filename']}\nSimilarity: {similarity:.4f}\nText: {result['text'][:200]}...")
        except Exception as e:
            st.error(f"Error computing similarities: {e}")

        # Frequency Distribution
        st.subheader("Word Frequency Distribution")
        plot_frequency_distribution(corpus_tokens)

        # Concordance
        st.subheader("Concordance")
        display_concordance(corpus_tokens, query)

if __name__ == '__main__':
    main()
