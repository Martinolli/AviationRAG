import streamlit as st
import json
import os
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

# Initialize OpenAI API
load_dotenv()

# Set up the OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load embeddings
def load_embeddings(filepath):
    with open(filepath, 'r') as file:
        return json.load(file)

def get_embedding(text, model="text-embedding-ada-002"):
    """
    Generate embedding using OpenAI's updated client.

    Args:
        text (str): The input text to generate an embedding for.
        model (str): The embedding model to use.

    Returns:
        list: The generated embedding.
    """
    try:
        response = client.Embedding.create(
            input=[text],
            model=model
        )
        # Extract the embedding correctly
        return response['data'][0]['embedding']
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

# Compute similarities
def compute_similarities(query_embedding, embeddings):
    similarities = []
    for embedding in embeddings:
        similarity = cosine_similarity([query_embedding], [embedding['embedding']])[0][0]
        similarities.append((embedding, similarity))
    return sorted(similarities, key=lambda x: x[1], reverse=True)

# Generate OpenAI response
def generate_response(context, query, model="gpt-4", temperature=0.7):
    prompt = f"The user asked: {query}\n\nHere is the context:\n{context}\n\nProvide a detailed response:"
    try:
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.choices[0].message['content']
    except Exception as e:
        return f"Error generating response: {e}"

# Streamlit app
def main():
    st.title("Aviation Knowledge Assistant")
    st.write("Explore aviation knowledge with conversational intelligence powered by embeddings and OpenAI.")

    # Sidebar options
    st.sidebar.header("Settings")
    model = st.sidebar.selectbox("Select OpenAI Model", ["gpt-3.5-turbo", "gpt-4"], index=1)
    temperature = st.sidebar.slider("Response Temperature", 0.0, 1.0, 0.7)

    # Load embeddings
    embeddings_file = st.sidebar.file_uploader("Upload Embeddings JSON", type="json")
    if embeddings_file is not None:
        embeddings = json.load(embeddings_file)
        st.sidebar.success("Embeddings loaded successfully!")

    # Conversation history
    if "history" not in st.session_state:
        st.session_state.history = []

    # User input
    user_query = st.text_input("Enter your query", key="query")
    if st.button("Submit Query") and user_query and embeddings_file:
        # Generate embedding for user query
        query_embedding = openai.embeddings.create(input=[user_query], model="text-embedding-ada-002")['data'][0]['embedding']
        
        # Compute similarities
        similarities = compute_similarities(query_embedding, embeddings)
        top_contexts = [sim[0]['text'] for sim in similarities[:5]]

        # Create context for response
        context = " ".join(top_contexts)

        # Generate response
        response = generate_response(context, user_query, model=model, temperature=temperature)
        st.session_state.history.append({"query": user_query, "response": response})

        # Display response and context
        st.subheader("Response")
        st.write(response)
        
        st.subheader("Top Relevant Contexts")
        for idx, context in enumerate(top_contexts, start=1):
            st.write(f"**Context {idx}:** {context}")

    # Display conversation history
    if st.session_state.history:
        st.subheader("Conversation History")
        for idx, item in enumerate(st.session_state.history, start=1):
            st.write(f"**Query {idx}:** {item['query']}")
            st.write(f"**Response {idx}:** {item['response']}")

# Run the app
if __name__ == "__main__":
    main()
