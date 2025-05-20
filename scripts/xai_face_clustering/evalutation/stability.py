import numpy as np
from sklearn.metrics import normalized_mutual_info_score

def evaluate_clustering_stability(clusterings):
    """
    Compute pairwise NMI between different cluster runs to estimate stability.

    Args:
        clusterings (List[np.ndarray]): List of cluster label arrays from repeated clustering.

    Returns:
        np.ndarray: NMI matrix (symmetrical)
    """
    n = len(clusterings)
    nmi_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            nmi = normalized_mutual_info_score(clusterings[i], clusterings[j])
            nmi_matrix[i, j] = nmi_matrix[j, i] = nmi

    np.fill_diagonal(nmi_matrix, 1.0)
    return nmi_matrix
