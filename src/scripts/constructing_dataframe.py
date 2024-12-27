import pandas as pd
import pickle
import json
import os

# Define paths
base_dir = r'C:\Users\Aspire5 15 i7 4G2050\ProjectRAG\AviationRAG'
corpus_path = os.path.join(base_dir, 'data', 'raw', 'aviation_corpus.pkl')
embeddings_path = os.path.join(base_dir, 'data', 'embeddings', 'aviation_embeddings.json')
output_path = os.path.join(base_dir, 'data', 'processed', 'combined_corpus_embeddings.csv')

# Load aviation corpus
with open(corpus_path, "rb") as f:
    corpus_data = pickle.load(f)

# Convert corpus data to DataFrame
df = pd.DataFrame(corpus_data)

# Load embeddings
with open(embeddings_path, 'r') as f:
    embeddings_data = json.load(f)

# Convert embeddings data to DataFrame
embeddings_df = pd.DataFrame(embeddings_data)

# Merge DataFrames based on 'filename' and 'text'
combined_df = pd.merge(df, embeddings_df, on=['filename', 'text'], how='left')

# Drop duplicate columns
combined_df = combined_df.loc[:, ~combined_df.columns.duplicated()]

# Save to CSV
combined_df.to_csv(output_path, index=False)

print(f"Combined data saved to CSV file: {output_path}")

# Print some statistics
print(f"Total rows in combined data: {len(combined_df)}")
print(f"Columns in combined data: {', '.join(combined_df.columns)}")
print(f"Number of rows with embeddings: {combined_df['embedding'].notna().sum()}")

