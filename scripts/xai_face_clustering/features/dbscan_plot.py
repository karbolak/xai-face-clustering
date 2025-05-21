# scripts/xai_face_clustering/features/dbscan_plot.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors

def plot_dbscan_kdistance(
    embeddings: np.ndarray,
    output_path: str,
    k: int = 5
):
    """
    Generate and save the k-distance plot for DBSCAN epsilon selection.

    Args:
        embeddings (np.ndarray): raw feature matrix (N, D).
        output_path (str): where to save the plot.
        k (int): which neighbor distance to plot (default=5).
    """
    # Standardize
    scaler = StandardScaler()
    X = scaler.fit_transform(embeddings)

    # Compute k-th neighbor distances
    nbrs = NearestNeighbors(n_neighbors=k)
    nbrs.fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = np.sort(distances[:, k - 1])

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(k_distances)
    plt.xlabel(f'Points sorted by distance to {k}th NN')
    plt.ylabel(f'{k}th NN distance')
    plt.title(f'k-Distance Graph (k={k})')
    plt.grid(True)
    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    plt.close()
    print(f"[INFO] Saved k-distance plot to {output_path}")
