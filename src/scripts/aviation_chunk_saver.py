import os
import json
import logging
import nltk
from nltk.tokenize import sent_tokenize
import tiktoken
import pickle

# Ensure necessary NLTK data is downloaded
nltk.download('punkt')

# Define absolute paths
base_dir = r'C:\Users\Aspire5 15 i7 4G2050\Project_2\AviationRAG'
pkl_file = os.path.join(base_dir, 'data', 'raw', 'aviation_corpus.pkl')
chunk_output_dir = os.path.join(base_dir, 'data', 'processed', 'chunked_documents')

# Set up logging
logging.basicConfig(level=logging.INFO, filename='chunking.log',
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Directory to save chunked JSON files
if not os.path.exists(chunk_output_dir):
    os.makedirs(chunk_output_dir)

# Initialize OpenAI tokenizer for accurate token counting
tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")

# Function to count tokens using OpenAI's tokenizer
def count_tokens(text):
    return len(tokenizer.encode(text))

# Function to chunk text by sentences and enforce token limits
def chunk_text_by_sentences(text, max_tokens=500, overlap=50):
    sentences = sent_tokenize(text)  # Tokenize into sentences
    chunks = []
    current_chunk = []
    current_tokens = 0

    for sentence in sentences:
        sentence_token_count = count_tokens(sentence)

        # Check if adding this sentence exceeds the max token limit
        if current_tokens + sentence_token_count > max_tokens:
            # Save the current chunk only if it's not empty
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            # Start a new chunk with overlap (only if not the first chunk)
            current_chunk = current_chunk[-overlap:] if overlap and len(chunks) > 0 else []
            current_tokens = count_tokens(" ".join(current_chunk))

        current_chunk.append(sentence)
        current_tokens += sentence_token_count

    # Add the last chunk if it exists
    if current_chunk:
        chunks.append(" ".join(current_chunk))

    # Validate and split oversized chunks
    return validate_and_split_chunks(chunks, max_tokens)

# Function to validate and split oversized chunks
def validate_and_split_chunks(chunks, max_tokens):
    """Ensure all chunks are within the token limit."""
    validated_chunks = []
    for chunk in chunks:
        token_count = count_tokens(chunk)
        if token_count > max_tokens:
            logging.warning(f"Chunk exceeds token limit: {token_count} tokens. Splitting further.")
            # Split the chunk into smaller parts
            words = chunk.split()
            temp_chunk = []
            temp_tokens = 0
            for word in words:
                word_token_count = count_tokens(word)
                if temp_tokens + word_token_count > max_tokens:
                    validated_chunks.append(" ".join(temp_chunk))
                    temp_chunk = []
                    temp_tokens = 0
                temp_chunk.append(word)
                temp_tokens += word_token_count
            if temp_chunk:
                validated_chunks.append(" ".join(temp_chunk))
        else:
            validated_chunks.append(chunk)
    return validated_chunks

# Function to process documents and save chunks as JSON
def save_documents_as_chunks(documents, output_dir, max_tokens=500, overlap=50):
    for doc in documents:
        if 'filename' not in doc or 'text' not in doc:
            logging.warning(f"Skipping document due to missing fields: {doc}")
            continue

        filename = doc['filename']
        text = doc['text']

        logging.info(f"Processing document: {filename}")

        # Chunk the text
        chunks = chunk_text_by_sentences(text, max_tokens, overlap)

        # Prepare JSON structure
        json_data = {
            "filename": filename,
            "chunks": [
                {"chunk_id": f"{os.path.splitext(filename)[0]}_chunk{i+1}", "text": chunk}
                for i, chunk in enumerate(chunks)
            ]
        }

        # Save JSON to file
        json_filename = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}.json")
        try:
            with open(json_filename, 'w', encoding='utf-8') as json_file:
                json.dump(json_data, json_file, ensure_ascii=False, indent=4)
            logging.info(f"Chunks saved for '{filename}' in '{json_filename}'")
        except Exception as e:
            logging.error(f"Failed to save chunks for '{filename}': {e}")

# Main routine
def main():
    # Load your PKL file containing documents
    if not os.path.exists(pkl_file):
        logging.error(f"Error: PKL file '{pkl_file}' not found!")
        return

    try:
        with open(pkl_file, 'rb') as file:
            documents = pickle.load(file)
        logging.info(f"Loaded {len(documents)} documents.")
    except Exception as e:
        logging.error(f"Failed to load PKL file: {e}")
        return

    # Process and save chunks for all documents
    save_documents_as_chunks(documents, chunk_output_dir)

    logging.info(f"All documents processed. Chunks saved in '{chunk_output_dir}'.")

if __name__ == '__main__':
    main()
