import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import os

# Load the JSON file
base_dir = r'C:\Users\Aspire5 15 i7 4G2050\Project\AviationRAG'
embeddings_path = os.path.join(base_dir, 'data', 'embeddings', 'aviation_embeddings.json')
with open('path/to/your/embeddings.json', 'r') as file:
    data = json.load(file)

# Assuming your embeddings are stored in a key like 'embeddings'
# and are a list of lists (or similar format)
embeddings = np.array(data['embeddings'])

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

# Or, using t-SNE
tsne = TSNE(n_components=2, random_state=42)
tsne_embeddings = tsne.fit_transform(embeddings)

plt.figure(figsize=(10, 6))
plt.scatter(tsne_embeddings[:, 0], tsne_embeddings[:, 1])
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE of Embeddings')
plt.show()
