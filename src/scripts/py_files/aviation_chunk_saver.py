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
base_dir = r'C:\Users\Aspire5 15 i7 4G2050\ProjectRAG\AviationRAG'
pkl_file = os.path.join(base_dir, 'data', 'raw', 'aviation_corpus.pkl')
chunk_output_dir = os.path.join(base_dir, 'data', 'processed', 'chunked_documents')
log_dir = os.path.join(base_dir, 'logs')  # Define the path to the logs folder

# Set up logging
log_file_path = os.path.join(log_dir, 'chunking.log')
logging.basicConfig(level=logging.INFO, filename=log_file_path,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Directory to save chunked JSON files
if not os.path.exists(chunk_output_dir):
    os.makedirs(chunk_output_dir)

# Initialize OpenAI tokenizer for accurate token counting
tokenizer = tiktoken.encoding_for_model("text-embedding-ada-002")

# Function to count tokens using OpenAI's tokenizer
def count_tokens(text):
    return len(tokenizer.encode(text))

# Function to check if a document has been processed already
def is_document_already_chunked(filename):
    """Checks if a chunk file already exists for a given document."""
    chunk_file = os.path.join(chunk_output_dir, f"{os.path.splitext(filename)[0]}_chunks.json")
    return os.path.exists(chunk_file)

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
        filename = doc['filename']

        # ✅ Skip document if it has already been processed
        if is_document_already_chunked(filename):
            logging.info(f"Skipping already chunked document: {filename}")
            print(f"Skipping already chunked document: {filename}")
            continue
        
        text = doc['text']
        metadata = doc.get('metadata', {})  # Get metadata if it exists, otherwise empty dict
        category = doc.get('category', '')  # Get category if it exists, otherwise empty string

        chunks = chunk_text_by_sentences(text, max_tokens, overlap)
        validated_chunks = validate_and_split_chunks(chunks, max_tokens)

        output_filename = os.path.join(output_dir, f"{os.path.splitext(filename)[0]}_chunks.json")
        
        print(f"Processing and saving chunks for {filename}")
        chunk_data = {
            "filename": filename,
            "metadata": metadata,
            "category": category,
            "chunks": [
                {
                    "chunk_id": f"{filename}_{i}",  # Add a unique chunk_id
                    "text": chunk,
                    "tokens": count_tokens(chunk)
                } for i, chunk in enumerate(validated_chunks)
            ]
        }
       
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(chunk_data, f, ensure_ascii=False, indent=2)

        logging.info(f"Processed and saved chunks for {filename}")

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
    print(f"All documents processed. Chunks saved in '{chunk_output_dir}'.")

if __name__ == '__main__':
    main()
