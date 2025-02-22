"""
This script does the following:

1. Plots the number of tokens for each document.
2. Shows the distribution of document categories.
3. Displays the top 20 most common words across all documents.
4. Illustrates the distribution of document lengths.
5. Plots the average number of personal names per document category.

This updated script includes:

1. A word cloud visualization of the most common words.
2. Use of Seaborn for enhanced bar plots and histograms.
3. A heatmap showing the frequency of the top 20 words across different categories.
The script will now generate six PNG files:

- token_counts.png
- category_distribution.png
- wordcloud.png
- doc_length_distribution.png
- avg_names_per_category.png
- word_category_heatmap.png

"""
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
from nltk import FreqDist
import numpy as np
from wordcloud import WordCloud
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from collections import Counter
from textblob import TextBlob
import textstat

# Define paths
base_dir = r'C:\Users\Aspire5 15 i7 4G2050\ProjectRAG\AviationRAG'
corpus_path = os.path.join(base_dir, 'data', 'raw', 'aviation_corpus.pkl')

# Create the directory if it doesn't exist
if not os.path.exists('pictures'):
    os.makedirs('pictures')


# importing the aviation corpus to extract the dictionary data

# Load aviation corpus
with open(corpus_path, "rb") as f:
    documents = pickle.load(f)

# generating a dataframe from the aviation corpus data

df_aviation_corpus = pd.DataFrame(documents)

print(df_aviation_corpus.head())


# Set up the plotting style
sns.set(style="whitegrid")

# 1. Plot number of tokens per document
doc_lengths = [len(doc['tokens']) for doc in documents]

# If the number of documents is too large, sample a subset
if len(doc_lengths) > 100:
    sampled_indices = np.random.choice(len(doc_lengths), 100, replace=False)
    doc_lengths = [doc_lengths[i] for i in sampled_indices]

plt.figure(figsize=(12, 6))
sns.lineplot(x=range(len(doc_lengths)), y=doc_lengths)
plt.title('Number of Tokens per Document')
plt.xlabel('Document Index')
plt.ylabel('Number of Tokens')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('assets/pictures/token_counts.png')
plt.close()

# 2. Plot distribution of document categories
categories = [doc['category'] for doc in documents]
category_counts = Counter(categories)
plt.figure(figsize=(12, 6))
sns.barplot(x=list(category_counts.keys()), y=list(category_counts.values()))
plt.title('Distribution of Document Categories')
plt.xlabel('Category')
plt.ylabel('Number of Documents')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('assets/pictures/category_distribution.png')
plt.close()

# 3. Create and save WordCloud of most common words
all_words = ' '.join([' '.join(doc['tokens']) for doc in documents])
wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_words)
plt.figure(figsize=(10, 5))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('Word Cloud of Most Common Words')
plt.tight_layout(pad=0)
plt.savefig('assets/pictures/wordcloud.png')
plt.close()

# 4. Plot distribution of document lengths
plt.figure(figsize=(12, 6))
sns.histplot(doc_lengths, bins=20, kde=True)
plt.title('Distribution of Document Lengths')
plt.xlabel('Number of Tokens')
plt.ylabel('Number of Documents')
plt.savefig('assets/pictures/doc_length_distribution.png')
plt.close()

# 5. Plot average number of personal names per document category
category_names = {}
for doc in documents:
    if doc['category'] not in category_names:
        category_names[doc['category']] = []
    category_names[doc['category']].extend(doc['personal_names'])

avg_names = {cat: len(names) / len([d for d in documents if d['category'] == cat]) 
             for cat, names in category_names.items()}

plt.figure(figsize=(12, 6))
sns.barplot(x=list(avg_names.keys()), y=list(avg_names.values()))
plt.title('Average Number of Personal Names per Document Category')
plt.xlabel('Category')
plt.ylabel('Average Number of Names')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('assets/pictures/avg_names_per_category.png')
plt.close()

# 6. Heatmap of top 20 words across categories
top_words = FreqDist(word for doc in documents for word in doc['tokens']).most_common(20)
word_freq_by_category = {cat: Counter() for cat in set(doc['category'] for doc in documents)}

for doc in documents:
    for word in doc['tokens']:
        if word in dict(top_words):
            word_freq_by_category[doc['category']][word] += 1

word_freq_matrix = np.array([[word_freq_by_category[cat][word] for word, _ in top_words] 
                             for cat in word_freq_by_category.keys()])

plt.figure(figsize=(15, 10))
sns.heatmap(word_freq_matrix, 
            xticklabels=[word for word, _ in top_words],
            yticklabels=list(word_freq_by_category.keys()),
            cmap='YlOrRd', annot=True, fmt='d')
plt.title('Heatmap of Top 20 Words Across Categories')
plt.xlabel('Words')
plt.ylabel('Categories')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('assets/pictures/word_category_heatmap.png')
plt.close()

from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()
sia_documents = [sia.polarity_scores(item["text"]) for item in documents]
avg_neg = [item['neg'] for item in sia_documents]
avg_neu = [item['neu'] for item in sia_documents]
sia_documents_compound = [item['compound'] for item in sia_documents]
avg_pos = [item['pos'] for item in sia_documents]


# Plotting the sentiment scores
# Create a DataFrame with the sentiment scores
sentiment_df = pd.DataFrame({
    'Negative': avg_neg,
    'Neutral': avg_neu,
    'Positive': avg_pos,
    'Compound': sia_documents_compound
})

# 1. Distribution of Compound Scores
plt.figure(figsize=(12, 6))
sns.histplot(data=sentiment_df, x='Compound', kde=True)
plt.title('Distribution of Compound Sentiment Scores')
plt.xlabel('Compound Score')
plt.ylabel('Count')
plt.savefig('assets/pictures/compound_score_distribution.png')
plt.close()

# 2. Boxplot of Sentiment Scores
plt.figure(figsize=(12, 6))
sns.boxplot(data=sentiment_df[['Negative', 'Neutral', 'Positive']])
plt.title('Boxplot of Sentiment Scores')
plt.ylabel('Score')
plt.savefig('assets/pictures/sentiment_scores_boxplot.png')
plt.close()

# 3. Heatmap of Sentiment Scores
plt.figure(figsize=(10, 8))
sns.heatmap(sentiment_df[['Negative', 'Neutral', 'Positive', 'Compound']].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap of Sentiment Scores')
plt.savefig('assets/pictures/sentiment_scores_heatmap.png')
plt.close()

# 4. Violin Plot of Sentiment Scores
plt.figure(figsize=(12, 6))
sns.violinplot(data=sentiment_df[['Negative', 'Neutral', 'Positive']])
plt.title('Violin Plot of Sentiment Scores')
plt.ylabel('Score')
plt.savefig('assets/pictures/sentiment_scores_violin.png')
plt.close()

# 5. Scatter Plot of Positive vs. Negative Scores
plt.figure(figsize=(10, 8))
sns.scatterplot(data=sentiment_df, x='Negative', y='Positive', hue='Compound', palette='viridis')
plt.title('Scatter Plot of Positive vs. Negative Scores')
plt.xlabel('Negative Score')
plt.ylabel('Positive Score')
plt.savefig('assets/pictures/positive_vs_negative_scatter.png')
plt.close()

# Fucntion to calculate the most frequent words
def most_frequent_words(text, n=5):
    tokens = word_tokenize(text)
    freq_dist = Counter(tokens)
    return freq_dist.most_common(n)

# Calculate readability score

def calculate_readability_score(text):
    return textstat.flesch_reading_ease(text)

# Calculate sentiment score

def calculate_sentiment_score(text):
    return TextBlob(text).sentiment.polarity

# Calculate Lexicon diversity score

def lexical_diversity(text):
    return (len(set(text)) / len(text))  * 100

# Add sentiment scores to df_aviation_corpus
df_aviation_corpus['Sentiment_Negative'] = sentiment_df['Negative']
df_aviation_corpus['Sentiment_Neutral'] = sentiment_df['Neutral']
df_aviation_corpus['Sentiment_Positive'] = sentiment_df['Positive']
df_aviation_corpus['Sentiment_Compound'] = sentiment_df['Compound']
df_aviation_corpus['texts_lengths'] = doc_lengths
df_aviation_corpus['frequent_words'] = df_aviation_corpus['text'].apply(most_frequent_words)
df_aviation_corpus['readability_score'] = df_aviation_corpus['text'].apply(calculate_readability_score)
df_aviation_corpus['sentiment_score'] = df_aviation_corpus['text'].apply(calculate_sentiment_score)
df_aviation_corpus['lexicon_diversity'] = df_aviation_corpus['tokens'].apply(lexical_diversity)


# Verify the new columns
print(df_aviation_corpus.head())

# Plot Lexical Diversity
# Boxplot
plt.figure(figsize=(12, 6))
sns.boxplot(data=df_aviation_corpus, y='lexicon_diversity')
plt.title('Boxplot of Lexical Diversity')
plt.ylabel('Lexical Diversity')
plt.tight_layout()
plt.savefig('assets/pictures/lexical_diversity_boxplot.png')
plt.close()

# Bar Plot (Average Lexical Diversity by Category)
plt.figure(figsize=(12, 6))
avg_lexical_diversity = df_aviation_corpus.groupby('category')['lexicon_diversity'].mean().reset_index()
sns.barplot(data=avg_lexical_diversity, x='category', y='lexicon_diversity')
plt.title('Average Lexical Diversity by Category')
plt.xlabel('Category')
plt.ylabel('Average Lexical Diversity')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('assets/pictures/avg_lexical_diversity_by_category.png')
plt.close()

# Violin Plot
plt.figure(figsize=(12, 6))
sns.violinplot(data=df_aviation_corpus, y='lexicon_diversity')
plt.title('Violin Plot of Lexical Diversity')
plt.ylabel('Lexical Diversity')
plt.tight_layout()
plt.savefig('assets/pictures/lexical_diversity_violin.png')
plt.close()

# Histogram
plt.figure(figsize=(12, 6))
sns.histplot(df_aviation_corpus['lexicon_diversity'], bins=20, kde=True)
plt.title('Histogram of Lexical Diversity')
plt.xlabel('Lexical Diversity')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('assets/pictures/lexical_diversity_histogram.png')
plt.close()

# If you want to save the updated dataframe
df_aviation_corpus.to_pickle('data/processed/updated_aviation_corpus.pkl')

df_aviation_corpus.to_csv("data/processed/updated_aviation_corpus.csv", index=False)

print(df_aviation_corpus.info())

print("Updated dataframe with sentiment scores has been created and saved.")

print("Sentiment analysis visualizations have been saved as PNG files.")

print("Analysis complete. Plots have been saved as PNG files.")

print(df_aviation_corpus.head())

print('Aviation Corpus: Filename, Texts Lengths, Frequent Words, Readability, Sentiment Scores, and Lexicon Diversity.')

print(df_aviation_corpus[['filename', 'texts_lengths', 'frequent_words', 'readability_score', 'sentiment_score','lexicon_diversity']].head())