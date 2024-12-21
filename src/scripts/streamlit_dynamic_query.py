import streamlit as st
import json
import os
from sklearn.metrics.pairwise import cosine_similarity
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
    try:
        with open(EMBEDDINGS_FILE, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        st.error(f"Embeddings file not found at {EMBEDDINGS_FILE}.")
        return None

def get_embedding(text, model="text-embedding-ada-002"):
    """Generate embedding using OpenAI's API."""
    try:
        response = client.embeddings.create(input=[text], model=model)
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

def generate_response(context, query):
    """Generate a detailed response using OpenAI."""
    prompt = f"""
    Context:
    {context}

    Question:
    {query}

    Provide a detailed and accurate response based on the context above.
    """
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"Error generating response: {e}")
        return None

def truncate_text(text, max_chars=300):
    """Truncate text to a specified number of characters."""
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def main():

    # Streamlit UI setup
    st.set_page_config(page_title="Aviation RAG Query Interface", layout="wide")
    st.title("Aviation RAG Query Interface")

    # Session state for query history and results
    if 'query_history' not in st.session_state:
        st.session_state['query_history'] = []

    if 'query_results' not in st.session_state:
        st.session_state['query_results'] = []

    # User query input
    query = st.text_input("Enter your query:", placeholder="e.g., Describe latent failures in aviation accidents")

    # Query submission
    if st.button("Submit Query"):
        if query:
            st.session_state.query_history.append(query)

            with st.spinner("Generating query embedding..."):
                query_embedding = get_embedding(query)

            if query_embedding:
                with st.spinner("Loading embeddings..."):
                    embeddings = load_embeddings()

                if embeddings:
                    with st.spinner("Computing similarities..."):
                        results = compute_similarity(embeddings, query_embedding)
                        st.session_state.query_results = results

                    # Combine context and generate response
                    context = "\n".join([result['text'] for result in results[:5]])
                    with st.spinner("Generating response..."):
                        response = generate_response(context, query)

                    # Display response
                    st.subheader("Generated Response")
                    if response:
                        st.write(response)
                    else:
                        st.error("Failed to generate a response.")

                    # Display results
                    st.subheader("Top Results")
                    for result in results[:5]:
                        st.markdown(f"**Chunk ID**: {result['chunk_id']}")
                        st.markdown(f"**Filename**: {result['filename']}")
                        st.markdown(f"**Similarity**: {result['similarity']:.4f}")
                        st.markdown(f"**Text**: {truncate_text(result['text'], max_chars=300)}")  # Truncated text
                        st.markdown("---")
                else:
                    st.error("Failed to load embeddings.")
            else:
                st.error("Failed to generate embedding for the query.")

    # Display query history
    if st.session_state['query_history']:
        st.sidebar.title("Query History")
        for i, past_query in enumerate(st.session_state['query_history'], 1):
            st.sidebar.write(f"{i}. {past_query}")

    # Display results history
    if st.session_state['query_results']:
        st.sidebar.title("Last Results")
        for result in st.session_state['query_results'][:3]:
            st.sidebar.write(f"Chunk ID: {result['chunk_id']} - {result['similarity']:.4f}")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Version 1.0 | Streamlit Interface**")

if __name__ == "__main__":
    main()
