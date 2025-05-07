import numpy as np
from xai_face_clustering.models.clustering import cluster_embeddings

def test_kmeans_labels():
    dummy_feats = np.random.rand(30, 50)
    labels = cluster_embeddings(dummy_feats, method="kmeans", n_clusters=2)
    assert len(labels) == 30
    assert all(label in [0, 1] for label in labels)