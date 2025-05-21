from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, adjusted_rand_score
import numpy as np

def cluster_embeddings(
    embeddings,
    method="kmeans",
    evaluate_stability=False,
    true_labels=None,
    **kwargs
):
    """
    Cluster embeddings with KMeans or DBSCAN and report metrics.

    Args:
        embeddings (np.ndarray): Feature vectors (N, D).
        method (str): "kmeans" or "dbscan".
        evaluate_stability (bool): Stability checks for KMeans only.
        true_labels (list or np.ndarray, optional): Ground truth labels for ARI.
        kwargs: Additional parameters:
            - KMeans: n_clusters (int)
            - DBSCAN: eps (float), min_samples (int)
    Returns:
        np.ndarray: Cluster assignments for each point.
    """
    # Select clustering model
    if method == "kmeans":
        n_clusters = kwargs.get("n_clusters", 2)
        model = KMeans(n_clusters=n_clusters, random_state=42)
    elif method == "dbscan":
        eps = kwargs.get("eps", 0.5)
        min_samples = kwargs.get("min_samples", 5)
        model = DBSCAN(eps=eps, min_samples=min_samples)
    else:
        raise ValueError(f"Unsupported clustering method: {method}")

    # Fit and predict cluster labels
    labels = model.fit_predict(embeddings)
    unique_labels = set(labels)

    # Report DBSCAN metrics
    if method == "dbscan":
        noise_count = list(labels).count(-1)
        cluster_ids = unique_labels - {-1}
        print(f"[INFO] DBSCAN found {len(cluster_ids)} clusters and {noise_count} noise points.")
        # Silhouette on non-noise if enough clusters
        if len(cluster_ids) >= 2:
            mask = labels != -1
            sil = silhouette_score(embeddings[mask], labels[mask])
            print(f"[INFO] Silhouette Score (excl. noise): {sil:.3f}")
        else:
            print("[WARN] Cannot compute silhouette score: need at least 2 clusters (excl. noise).")
        # ARI vs ground truth if available
        if true_labels is not None:
            ari = adjusted_rand_score(true_labels, labels)
            print(f"[INFO] Adjusted Rand Index vs ground truth: {ari:.3f}")

    # Report KMeans metrics
    else:
        sil = silhouette_score(embeddings, labels)
        print(f"[INFO] Silhouette Score: {sil:.3f}")
        if true_labels is not None:
            ari = adjusted_rand_score(true_labels, labels)
            print(f"[INFO] Adjusted Rand Index vs ground truth: {ari:.3f}")

    # Stability for KMeans
    if evaluate_stability and method == "kmeans":
        print("[INFO] Evaluating clustering stability over multiple runs...")
        scores = []
        for seed in range(5, 10):
            km = KMeans(n_clusters=kwargs.get("n_clusters", 2), random_state=seed)
            alt_labels = km.fit_predict(embeddings)
            ari_seed = adjusted_rand_score(labels, alt_labels)
            scores.append(ari_seed)
        print(f"[INFO] Avg Adjusted Rand Index (5 runs): {np.mean(scores):.3f}")

    return labels
