import numpy as np
from sklearn.linear_model import LogisticRegression
from scripts.xai_face_clustering.models.xai import run_shap_explanation

def test_shap_explanation_runs(tmp_path):
    X = np.random.rand(10, 20)
    y = [0]*5 + [1]*5
    model = LogisticRegression().fit(X, y)
    run_shap_explanation(model, X, y, num_examples=2)
    # Should not crash; files are saved to disk
