import numpy as np
import joblib
import matplotlib.pyplot as plt
import os

EMBED_PATH = "xai_face_clustering/features/embeddings.npz"
SVM_PATH   = "xai_face_clustering/models/surrogate_model.joblib"
PCA_PATH   = "xai_face_clustering/models/pca_model.joblib"
CLUSTER_MAP_PATH = "xai_face_clustering/models/cluster_label_map.json"

data = np.load(EMBED_PATH, allow_pickle=True)
X = data['embeddings']
y_true = np.array(data['labels'])  # 0=Real, 1=AI

pca_model = joblib.load(PCA_PATH)
X_pca = pca_model.transform(X)  # shape: (N, n_pca_components)

svm = joblib.load(SVM_PATH)
if X_pca.shape[1] != svm.n_features_in_:
    raise ValueError(f"PCA output shape {X_pca.shape[1]} does not match SVM input {svm.n_features_in_}.")

cluster_pred = svm.predict(X_pca)

if os.path.exists(CLUSTER_MAP_PATH):
    import json
    with open(CLUSTER_MAP_PATH, 'r') as f:
        cluster_map = json.load(f)
    
    mapped_labels = np.array([cluster_map.get(str(c), -1) for c in cluster_pred])
else:
    mapped_labels = cluster_pred

# project for 2D visualization (another PCA for 2D)
from sklearn.decomposition import PCA
pca_vis = PCA(n_components=2)
X_vis = pca_vis.fit_transform(X_pca)  # else X for "raw" 2D projection

# plot true Label vs SVM-cluster assignment
plt.figure(figsize=(10,8))
markers = {0: 'o', 1: 's'}  # o for Real, s for AI
colors = ['tab:blue', 'tab:orange']

for true_class, label_str, color in zip([0, 1], ['Real', 'AI'], colors):
    for cluster in np.unique(cluster_pred):
        mask = (y_true == true_class) & (cluster_pred == cluster)
        plt.scatter(
            X_vis[mask, 0],
            X_vis[mask, 1],
            label=f"True: {label_str}, SVM cluster: {cluster}",
            alpha=0.5,
            marker=markers[true_class],
            edgecolor='k',
            linewidths=0.2,
            s=25,
            c=color if cluster == 0 else None  
        )

plt.title("SVM Surrogate Cluster Assignments vs. Ground Truth")
plt.xlabel("PCA Component 1")
plt.ylabel("PCA Component 2")
plt.legend(fontsize="small", loc="best", frameon=True, ncol=1)
plt.tight_layout()
plt.show()

# optional alternative to highlight where predicted class != true
misclassified = mapped_labels != y_true
if np.any(misclassified):
    plt.figure(figsize=(10,8))
    plt.scatter(X_vis[~misclassified,0], X_vis[~misclassified,1], c='gray', alpha=0.3, s=15, label='Correct')
    plt.scatter(X_vis[misclassified,0], X_vis[misclassified,1], c='red', alpha=0.6, s=25, label='SVM≠True Label')
    plt.title("Misclassified Points by SVM Surrogate Mapping")
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.legend()
    plt.tight_layout()
    plt.show()
else:
    print("No misclassifications found (mapped_labels == y_true everywhere).")
