from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

def train_surrogate_model(X_train, y_train, X_test, y_test, method="logreg"):
    """
    Train a simple surrogate classifier to mimic clustering decisions.
    
    This classifier treats cluster ID as the target label.
    Used as input to SHAP or LIME for feature importance explanations.
    We print a basic classification report as a sanity check -- high accuracy implies good mimicry of clustering.

    Args:
        X (np.ndarray): PCA-reduced feature matrix (N, D)
        cluster_labels (np.ndarray): Cluster assignments (N,)
        method (str): 'logreg' or 'tree'

    Returns:
        Trained sklearn model
    """
    if method == "logreg":
        model = LogisticRegression(max_iter=1000)
    elif method == "tree":
        model = DecisionTreeClassifier(max_depth=3)
    else:
        raise ValueError(f"Unsupported surrogate model: {method}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("[INFO] Surrogate model test performance:")
    print(classification_report(y_test, y_pred))

    return model