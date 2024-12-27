import os
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from collections import Counter
from textblob import TextBlob
import textstat

# Function Definitions (Same as before)
def calculate_token_length(text):
    tokens = word_tokenize(text)
    return len(tokens)

def calculate_readability_score(text):
    return textstat.flesch_reading_ease(text)

def most_frequent_words(text, n=5):
    tokens = word_tokenize(text)
    freq_dist = Counter(tokens)
    return freq_dist.most_common(n)

def calculate_sentiment_score(text):
    return TextBlob(text).sentiment.polarity

def calculate_embedding_norm(embedding):
    return np.linalg.norm(embedding) if embedding is not None else None

def calculate_token_density(token_count, char_length):
    return token_count / char_length if char_length > 0 else 0

def calculate_similarity_to_baseline(embedding, baseline_embedding):
    return cosine_similarity([embedding], [baseline_embedding])[0][0] if embedding is not None else None

# Load Data
aviation_corpus_file_path = "data/processed/aviation_corpus.json"
embeddings_file_path = "data/embeddings/aviation_embeddings.json"

corpus_df = pd.read_json(aviation_corpus_file_path)
embeddings_df = pd.read_json(embeddings_file_path)

# Ensure both dataframes include `filename` to establish the relationship
if "filename" not in corpus_df.columns or "filename" not in embeddings_df.columns:
    raise ValueError("Both chunks and embeddings must include a 'filename' column to link documents.")

# Merge chunks and embeddings on `chunk_id`
merged_df = pd.merge(corpus_df, embeddings_df, on="chunk_id")

# Compute Metrics
baseline_embedding = np.random.rand(len(merged_df["embedding"].iloc[0]))  # Replace with actual baseline

merged_df["token_count"] = merged_df["text"].apply(calculate_token_length)
merged_df["readability_score"] = merged_df["text"].apply(calculate_readability_score)
merged_df["frequent_words"] = merged_df["text"].apply(most_frequent_words)
merged_df["sentiment_score"] = merged_df["text"].apply(calculate_sentiment_score)
merged_df["embedding_norm"] = merged_df["embedding"].apply(calculate_embedding_norm)
merged_df["similarity_to_baseline"] = merged_df["embedding"].apply(lambda emb: calculate_similarity_to_baseline(emb, baseline_embedding))
merged_df["char_length"] = merged_df["text"].apply(len)
merged_df["token_density"] = merged_df.apply(lambda row: calculate_token_density(row["token_count"], row["char_length"]), axis=1)

# Save Detailed DataFrame
detailed_metrics_path = "data/processed/detailed_metrics_with_docs.csv"
merged_df.to_csv(detailed_metrics_path, index=False)

print(f"Detailed metrics saved to {detailed_metrics_path}.")
