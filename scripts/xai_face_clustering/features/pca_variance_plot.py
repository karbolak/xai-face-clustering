import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def plot_pca_variance(embeddings: np.ndarray, output_path: str):
    """Generate and save PCA explained variance plot."""
    #standardize 
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)

    #fit PCA
    pca = PCA()
    pca.fit(scaled)

    plt.figure(figsize=(10, 6))
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('PCA - Cumulative Explained Variance')
    plt.grid(True)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"[INFO] Saved PCA explained variance plot to {output_path}")
