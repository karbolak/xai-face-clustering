import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

def plot_dbscan_clusters(embeddings: np.ndarray, output_path: str, eps=0.5, min_samples=5):
    """Generate and save DBSCAN clustering plot."""
    # Standardize features
    scaler = StandardScaler()
    scaled = scaler.fit_transform(embeddings)

    # Fit DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(scaled)

    # Plot clusters
    plt.figure(figsize=(10, 6))
    unique_labels = set(labels)
    for label in unique_labels:
        label_mask = (labels == label)
        color = 'k' if label == -1 else plt.cm.nipy_spectral(float(label) / len(unique_labels))
        plt.scatter(scaled[label_mask, 0], scaled[label_mask, 1], c=[color], label=f'Cluster {label}')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('DBSCAN Clustering')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path)
    print(f"[INFO] Saved DBSCAN clustering plot to {output_path}")

