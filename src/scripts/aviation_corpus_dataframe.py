import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from collections import Counter
from textblob import TextBlob
import textstat

# Function to calculate token length
def calculate_token_length(text):
    tokens = word_tokenize(text)
    return len(tokens)

# Function to calculate readability score
def calculate_readability_score(text):
    return textstat.flesch_reading_ease(text)

# Function to calculate most frequent words
def most_frequent_words(text, n=5):
    tokens = word_tokenize(text)
    freq_dist = Counter(tokens)
    return freq_dist.most_common(n)

# Function to calculate sentiment score
def calculate_sentiment_score(text):
    return TextBlob(text).sentiment.polarity

# Function to calculate embedding norm
def calculate_embedding_norm(embedding):
    return np.linalg.norm(embedding) if embedding is not None else None

# Function to calculate token density
def calculate_token_density(token_count, char_length):
    return token_count / char_length if char_length > 0 else 0

# Function to calculate similarity to a baseline embedding
def calculate_similarity_to_baseline(embedding, baseline_embedding):
    return cosine_similarity([embedding], [baseline_embedding])[0][0] if embedding is not None else None

# Load your existing data
chunk_file_path = "data/processed/chunked_documents/chunked_aviation_corpus.json"
embeddings_file_path = "data/embeddings/aviation_embeddings.json"

# Load data
chunks_df = pd.read_json(chunk_file_path)
embeddings_df = pd.read_json(embeddings_file_path)

# Merge chunks and embeddings dataframes
merged_df = pd.merge(chunks_df, embeddings_df, on="chunk_id")

# Compute metrics
baseline_embedding = np.random.rand(len(merged_df["embedding"].iloc[0]))  # Replace with an actual baseline

merged_df["token_count"] = merged_df["text"].apply(calculate_token_length)
merged_df["readability_score"] = merged_df["text"].apply(calculate_readability_score)
merged_df["frequent_words"] = merged_df["text"].apply(most_frequent_words)
merged_df["sentiment_score"] = merged_df["text"].apply(calculate_sentiment_score)
merged_df["embedding_norm"] = merged_df["embedding"].apply(calculate_embedding_norm)
merged_df["similarity_to_baseline"] = merged_df["embedding"].apply(lambda emb: calculate_similarity_to_baseline(emb, baseline_embedding))
merged_df["char_length"] = merged_df["text"].apply(len)
merged_df["token_density"] = merged_df.apply(lambda row: calculate_token_density(row["token_count"], row["char_length"]), axis=1)

# Save the detailed DataFrame
detailed_metrics_path = "data/processed/detailed_metrics.csv"
merged_df.to_csv(detailed_metrics_path, index=False)

print(f"Detailed metrics saved to {detailed_metrics_path}.")
