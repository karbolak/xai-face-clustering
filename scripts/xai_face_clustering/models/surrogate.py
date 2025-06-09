from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

import os, joblib

SURROGATE_SAVE_PATH = "scripts/xai_face_clustering/models/surrogate_model.joblib"

def train_surrogate_model(X_train, y_train, X_test, y_test, method="logreg"):
    """
    Train a simple surrogate classifier to mimic clustering decisions.

    This classifier treats cluster ID as the target label.
    Used as input to SHAP or LIME for feature importance explanations.
    We print a basic classification report as a sanity check -- high accuracy implies good mimicry of clustering.

    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels (e.g., cluster IDs)
        X_test (np.ndarray): Test features
        y_test (np.ndarray): Test labels
        method (str): 'logreg', 'tree', or 'svm'

    Returns:
        Trained sklearn model
    """
    if len(set(y_train)) < 2:
        print("[ERROR] Surrogate model training requires at least 2 classes.")
        return None

    if method == "logreg":
        model = LogisticRegression(max_iter=1000)
    elif method == "tree":
        model = DecisionTreeClassifier(max_depth=3)
    elif method == "svm":
        model = SVC(kernel="sigmoid", probability=True)  # kernel can be 'linear', 'poly', 'rbf', 'sigmoid'
    else:
        raise ValueError(f"Unsupported surrogate model: {method}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("[INFO] Surrogate model test performance:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    joblib.dump(model, SURROGATE_SAVE_PATH)
    print(f"[INFO] Saved surrogate model to {SURROGATE_SAVE_PATH}")

    return model
