from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
import numpy as np

def cluster_embeddings(embeddings, method="kmeans", evaluate_stability=False, **kwargs):
    """
    Cluster embeddings using the specified method and optionally evaluate clustering stability.

    Args:
        embeddings (np.ndarray): Feature vectors (N, D).
        method (str): "kmeans" or "dbscan".
        evaluate_stability (bool): Whether to perform repeated clustering for stability.
        kwargs: Additional clustering parameters.

    Returns:
        cluster_labels (np.ndarray): Cluster assignments for each point.
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

    if len(set(cluster_labels)) > 1 and -1 not in set(cluster_labels):
        score = silhouette_score(embeddings, cluster_labels)
        print(f"[INFO] Silhouette Score: {score:.3f}")
    else:
        print("[WARN] Cannot compute silhouette score: only one cluster or noise detected.")

    if evaluate_stability and method == "kmeans":
        print("[INFO] Evaluating clustering stability over multiple runs...")
        scores = []
        for seed in range(5, 10):
            km = KMeans(n_clusters=k, random_state=seed)
            labels_alt = km.fit_predict(embeddings)
            ari = adjusted_rand_score(cluster_labels, labels_alt)
            scores.append(ari)
        avg_ari = np.mean(scores)
        print(f"[INFO] Avg Adjusted Rand Index (5 runs): {avg_ari:.3f}")

    return cluster_labels
