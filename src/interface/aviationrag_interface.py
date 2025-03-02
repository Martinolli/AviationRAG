import streamlit as st
import json
import pandas as pd
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

# Set up OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load embeddings
@st.cache_data(ttl=3600, show_spinner=False)
def load_embeddings(embeddings_file):
    with open(embeddings_file, "r", encoding="utf-8") as file:
        return json.load(file)

def get_query_embedding(query, model="text-embedding-ada-002"):
    """Generate embedding for user query."""
    response = client.embeddings.create(input=[query], model=model)
    return response.data[0].embedding

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    return sum(a * b for a, b in zip(vec1, vec2)) / (sum(a ** 2 for a in vec1) ** 0.5 * sum(b ** 2 for b in vec2) ** 0.5)

# Retrieve top results
def retrieve_top_chunks(query_embedding, embeddings_data, top_n=5):
    results = []
    for chunk in embeddings_data:
        similarity = cosine_similarity(query_embedding, chunk["embedding"])
        results.append({
            "Filename": chunk["filename"],
            "Chunk ID": chunk["chunk_id"],
            "Similarity": similarity,
            "Text": chunk["text"]
        })
    return sorted(results, key=lambda x: x["Similarity"], reverse=True)[:top_n]

# Streamlit UI
def main():
    st.title("AviationRAG: Query Interface")
    
    # Load embeddings
    embeddings_file = "data/embeddings/aviation_embeddings.json"
    embeddings_data = load_embeddings(embeddings_file)
    
    # User input
    query = st.text_input("Enter your query:", "")
    
    if st.button("Submit Query"):
        if query:
            st.info("Generating query embedding...")
            query_embedding = get_query_embedding(query)
            
            st.info("Retrieving top chunks...")
            top_chunks = retrieve_top_chunks(query_embedding, embeddings_data)
            
            # Display results
            st.subheader("Top Results:")
            for chunk in top_chunks:
                st.markdown(f"**Filename:** {chunk['Filename']}")
                st.markdown(f"**Chunk ID:** {chunk['Chunk ID']}")
                st.markdown(f"**Similarity:** {chunk['Similarity']:.4f}")
                st.markdown(f"**Text:** {chunk['Text']}")
                st.markdown("---")
                
            st.success("Response generated successfully!")
        else:
            st.warning("Please enter a query.")

if __name__ == "__main__":
    main()
