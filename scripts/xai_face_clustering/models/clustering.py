from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np

def cluster_embeddings(embeddings, method="kmeans", **kwargs):
    """
        Clusters embeddings using the specified method.
        
        Also:
            Silhouette score is a good sanity check for clustering quality.
            For DBSCAN:
                Points labeled -1 are considered noise.
                Parameters like n_clusters, eps, and min_samples are passed from main.py.

        Args:
            embeddings (np.ndarray): (N, D) array of feature vectors.
            method (str): "kmeans" or "dbscan"
            kwargs: clustering-specific parameters.

        Returns:
            cluster_labels (np.ndarray): Array of cluster assignments (shape: N,)
    """
    if method == "kmeans":
        k = kwargs.get("n_clusters", 2)
        model = KMeans(n_clusters=k, random_state=42)
    elif method == "dbscan":
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    cluster_labels = model.fit_predict(embeddings)

    # Optional: print silhouette score if labels are valid
    if len(set(cluster_labels)) > 1 and -1 not in set(cluster_labels):
        score = silhouette_score(embeddings, cluster_labels)
        print(f"[INFO] Silhouette Score: {score:.3f}")
    else:
        print("[WARN] Cannot compute silhouette score: only 1 cluster or noise points present.")

    return cluster_labels
