import numpy as np
import joblib
import shap
import os

EMBEDDINGS_CACHE = "scripts/xai_face_clustering/features/embeddings.npz"
SCALER_PATH = "scripts/xai_face_clustering/models/scaler.joblib"
PCA_PATH = "scripts/xai_face_clustering/models/pca_model.joblib"
SHAP_BG_PATH = "scripts/xai_face_clustering/models/shap_background.npz"

data = np.load(EMBEDDINGS_CACHE, allow_pickle=True)
X = data["embeddings"]

scaler = joblib.load(SCALER_PATH)
pca = joblib.load(PCA_PATH)

X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)

K = 20
bg_summary = shap.kmeans(X_pca, K)

os.makedirs(os.path.dirname(SHAP_BG_PATH), exist_ok=True)
np.savez(SHAP_BG_PATH, background=bg_summary.data)

print(f"SHAP background saved to {SHAP_BG_PATH}")
print(f"Shape: {bg_summary.data.shape}")
