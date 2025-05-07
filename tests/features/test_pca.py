import numpy as np
from xai_face_clustering.features.pca import apply_pca

def test_pca_output_dim():
    dummy_feats = np.random.rand(50, 2048)
    reduced = apply_pca(dummy_feats, n_components=50, fit=True, save=False)
    assert reduced.shape == (50, 50)
