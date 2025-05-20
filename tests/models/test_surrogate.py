import numpy as np
from scripts.xai_face_clustering.models.surrogate import train_surrogate_model

def test_surrogate_fit_predict():
    X = np.random.rand(25, 60)
    y = [0]*12 + [1]*13
    model = train_surrogate_model(X, y, method="logreg")
    preds = model.predict(X)
    assert len(preds) == 25
