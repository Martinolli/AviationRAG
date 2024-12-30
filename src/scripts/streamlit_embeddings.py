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
import pandas as pd

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
@st.cache_data
def load_embeddings():
    with open('data/embeddings/aviation_embeddings.json', 'r') as f:
        embeddings = json.load(f)
    return embeddings

# Load corpus
@st.cache_data
def load_corpus():
    with open('data/raw/aviation_corpus.pkl', 'rb') as f:
        corpus = pickle.load(f)
    return corpus

# Similarity search
def similarity_search(embeddings, query_embedding):
    embeddings_array = np.array([item['embedding'] for item in embeddings])
    similarities = cosine_similarity([query_embedding], embeddings_array).flatten()
    for i, item in enumerate(embeddings):
        item['similarity'] = similarities[i]
    sorted_results = sorted(embeddings, key=lambda x: x['similarity'], reverse=True)
    return sorted_results[:10]

# Display frequency distribution
def plot_frequency_distribution(corpus_tokens):
    fdist = FreqDist(corpus_tokens)
    top_words = fdist.most_common(20)
    words, counts = zip(*top_words)
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts)
    plt.title('Top 20 Words in Corpus')
    plt.xticks(rotation=70, ha='right')
    st.pyplot(plt)

# Generate concordance
def generate_concordance(corpus_tokens, word):
    text = Text(corpus_tokens)
    concordance_results = text.concordance_list(word, width=80)
    return concordance_results

def main():
    st.title("Aviation RAG Analysis Tool")

    # Sidebar info
    st.sidebar.header("Corpus Insights")
    corpus = load_corpus()
    all_tokens = [token for doc in corpus for token in doc['tokens']]
    st.sidebar.write(f"Total Documents: {len(corpus)}")
    st.sidebar.write(f"Total Tokens: {len(all_tokens)}")

    # Query input
    query_text = st.text_input("Enter your query text:")
    if query_text:
        # Generate query embedding (mock embedding for example purposes)
        st.info("Generating query embedding...")
        query_embedding = get_embedding(query_text)
        if query_embedding is None:
            st.error("Failed to generate query embedding.")
            return
        
        # Load embeddings
        embeddings = load_embeddings()

        # Similarity search
        results = similarity_search(embeddings, query_embedding)
        st.subheader("Top Similarity Results")
        for result in results:
            st.write(f"Chunk ID: {result['chunk_id']}")
            st.write(f"Filename: {result['filename']}")
            st.write(f"Similarity: {result['similarity']:.4f}")
            st.write(f"Text Snippet: {result['text'][:200]}...")

        # Similarity visualization
        st.subheader("Similarity Scores Visualization")
        if results:
            df = pd.DataFrame({
                'Chunk ID': [res['chunk_id'] for res in results],
                'Similarity': [res['similarity'] for res in results]
            })
            st.bar_chart(df.set_index('Chunk ID'))

    # Corpus analysis
    st.subheader("Corpus Analysis")
    analysis_type = st.radio("Choose analysis type:", ("Frequency Distribution", "Concordance"))
    if analysis_type == "Frequency Distribution":
        st.write("### Frequency Distribution")
        plot_frequency_distribution(all_tokens)
    elif analysis_type == "Concordance":
        word = st.text_input("Enter a word for concordance analysis:")
        if word:
            concordance_results = generate_concordance(all_tokens, word)
            st.write(f"Concordance for '{word}':")
            for entry in concordance_results:
                st.text(f"... {entry.left_print} {entry.query} {entry.right_print} ...")

if __name__ == "__main__":
    main()
