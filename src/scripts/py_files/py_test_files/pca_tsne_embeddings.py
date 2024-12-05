import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import os

# Load the JSON file
base_dir = r'C:\Users\Aspire5 15 i7 4G2050\Project\AviationRAG'
embeddings_path = os.path.join(base_dir, 'data', 'embeddings', 'aviation_embeddings.json')
with open(embeddings_path, 'r') as file:
    data = json.load(file)

# Assuming your embeddings are stored in a key like 'embeddings'
# Print the structure of the loaded data
print("Structure of loaded data:")
print(type(data))
if isinstance(data, dict):
    print("Keys in the dictionary:", data.keys())
elif isinstance(data, list):
    print("Number of items in the list:", len(data))
    if len(data) > 0:
        print("Type of the first item:", type(data[0]))
        if isinstance(data[0], dict):
            print("Keys in the first item:", data[0].keys())

# Extract embeddings from the list of dictionaries
embeddings = np.array([item['embedding'] for item in data if item['filename'] == '14cfr_safety_management_systems.pdf'])

# Print some information about the embeddings
print(f"Number of embeddings: {len(embeddings)}")
print(f"Dimension of each embedding: {len(embeddings[0])}")

# Choose either PCA or t-SNE for visualization

# Using PCA
pca = PCA(n_components=2)
pca_embeddings = pca.fit_transform(embeddings)

plt.figure(figsize=(10, 6))
plt.scatter(pca_embeddings[:, 0], pca_embeddings[:, 1])
plt.xlabel('PCA Component 1')
plt.ylabel('PCA Component 2')
plt.title('PCA of Embeddings')
plt.show()


# Assuming embeddings is your PCA or t-SNE reduced data
kmeans = KMeans(n_clusters=1)  # Choose an appropriate number of clusters
labels = kmeans.fit_predict(pca_embeddings)

silhouette_avg = silhouette_score(pca_embeddings, labels)
print(f"Silhouette Score: {silhouette_avg}")

# Using t-SNE
tsne = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200, n_iter=1000)
tsne_embeddings = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 6))
plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1])
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE of Embeddings')
plt.show()

