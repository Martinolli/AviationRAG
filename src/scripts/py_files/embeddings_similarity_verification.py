import json
import os
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up the OpenAI API key
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

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
        response = client.embeddings.create(
            input=[text],
            model=model
        )
        return response.data[0].embedding
    except Exception as e:
        print(f"Error generating embedding: {e}")
        return None

def load_embeddings(file_path, batch_size=1000):
    """
    Load embeddings from a JSON file in batches to improve performance.

    Args:
        file_path (str): Path to the JSON file containing embeddings.
        batch_size (int): Number of embeddings to load at a time.

    Returns:
        generator: Generator yielding embeddings in batches.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)

    for i in range(0, len(data), batch_size):
        yield data[i:i + batch_size]

def compute_cosine_similarity(query_embedding, embeddings):
    """
    Compute cosine similarity between a query embedding and a list of embeddings.

    Args:
        query_embedding (list): The embedding for the query.
        embeddings (list): List of embeddings to compare against.

    Returns:
        list: List of cosine similarity scores.
    """
    query_vector = np.array(query_embedding).reshape(1, -1)
    embedding_vectors = np.array([item['embedding'] for item in embeddings])

    return cosine_similarity(query_vector, embedding_vectors)[0]

def filter_and_rank_embeddings(embeddings, similarities, top_n=15):
    """
    Filter and rank embeddings based on similarity scores.

    Args:
        embeddings (list): List of embeddings with metadata.
        similarities (list): Corresponding similarity scores.
        top_n (int): Number of top results to return.

    Returns:
        list: Top N ranked embeddings with metadata and similarity scores.
    """
    results = [
        {
            'chunk_id': emb['chunk_id'],
            'filename': emb['filename'],
            'text': emb['text'],
            'similarity': sim
        }
        for emb, sim in zip(embeddings, similarities)
    ]

    # Sort results by similarity
    results = sorted(results, key=lambda x: x['similarity'], reverse=True)
    return results[:top_n]

def generate_response(context, query, model):
    """
    Generate a response using OpenAI.

    Args:
        context (str): The context string generated from retrieved chunks.
        query (str): The user query.
        model (str): The model to use for generating a response.

    Returns:
        str: The generated response from OpenAI.
    """
    max_context_length = 8000  # Adjust based on the model's capabilities
    truncated_context = context[:max_context_length]
    
    prompt = f"""
    Context:
    {truncated_context}

    Question:
    {query}

    Provide a detailed, comprehensive, and accurate response based on the context above. 
    Include relevant facts, explanations, and examples where appropriate. 
    For each key piece of information in your response, cite the source document in square brackets, 
    e.g., [Document: Safety Manual]. If information comes from multiple sources, list all relevant sources.
    If the context doesn't contain enough information to fully answer the question, 
    clearly state what information is missing or uncertain.
    """
    try:
        if model in ["gpt-3.5-turbo", "gpt-4"]:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5
            )
            return response.choices[0].message.content.strip()
        elif model == "text-davinci-003":
            response = client.completions.create(
                model=model,
                prompt=prompt,
                temperature=0.4,
                max_tokens=500
            )
            return response.choices[0].text.strip()
        else:
            raise ValueError(f"Unsupported model: {model}")
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

def generate_followup_questions(response, query, model):
    prompt = f"""
    Based on the following query and response, generate 2-3 relevant follow-up questions 
    that would help provide a more comprehensive answer:

    Original Query: {query}

    Response: {response}

    Follow-up Questions:
    """
    try:
        followup_response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7
        )
        return followup_response.choices[0].message.content.strip().split("\n")
    except Exception as e:
        print(f"Error generating follow-up questions: {e}")
        return []

def fact_check_response(response, context, model):
    prompt = f"""
    Response to be fact-checked:
    {response}

    Original Context:
    {context}

    Please fact-check the response against the original context. 
    Identify any inaccuracies, misinterpretations, or unsupported claims. 
    If the response is accurate and well-supported, confirm this.
    Provide a brief report on the accuracy and completeness of the response.
    """
    try:
        fact_check_result = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2
        )
        return fact_check_result.choices[0].message.content.strip()
    except Exception as e:
        print(f"Error during fact-checking: {e}")
        return "Fact-checking could not be completed due to an error."

if __name__ == "__main__":
    EMBEDDINGS_FILE = "data/embeddings/aviation_embeddings.json"

    QUERY_TEXT = input("Enter your query text: ")
    print("Choose a model: 1. gpt-3.5-turbo  2. gpt-4  3. text-davinci-003")
    MODEL_SELECTION = input("Enter model number (1/2/3): ")
    MODEL_MAP = {"1": "gpt-3.5-turbo", "2": "gpt-4", "3": "text-davinci-003"}
    MODEL = MODEL_MAP.get(MODEL_SELECTION, "gpt-4")  # Default to GPT-4

    TOP_N = 15

    try:
        print("Generating query embedding...")
        query_embedding = get_embedding(QUERY_TEXT)
        if query_embedding is None:
            raise ValueError("Failed to generate query embedding")

        print("Loading embeddings...")
        top_results = []
        for batch in load_embeddings(EMBEDDINGS_FILE):
            print(f"Processing batch of {len(batch)} embeddings...")
            similarities = compute_cosine_similarity(query_embedding, batch)
            top_results.extend(filter_and_rank_embeddings(batch, similarities, top_n=TOP_N))

        unique_texts = set()
        combined_context = ""
        for result in sorted(top_results, key=lambda x: x['similarity'], reverse=True)[:TOP_N]:
            if result['text'] not in unique_texts:
                unique_texts.add(result['text'])
                combined_context += f"{result['text']}\n"

        print("Generating response...")
        response = generate_response(combined_context, QUERY_TEXT, MODEL)

        print("\nGenerated Response:")
        print(response)

        print("\nFact-Checking Response...")
        fact_check_report = fact_check_response(response, combined_context, MODEL)
        print("Fact-Check Report:")
        print(fact_check_report)

        print("\nGenerating Follow-up Questions...")
        followup_questions = generate_followup_questions(response, QUERY_TEXT, MODEL)
        for i, question in enumerate(followup_questions, 1):
            print(f"\nFollow-up Question {i}: {question}")
            followup_response = generate_response(combined_context, question, MODEL)
            print(f"Follow-up Response {i}:")
            print(followup_response)
  
    except Exception as e:
        print(f"Error: {e}")