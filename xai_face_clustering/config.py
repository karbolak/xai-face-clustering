from pathlib import Path

# project_root/xai_face_clustering/config.py
BASE_DIR     = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR.parent.parent / "artifacts"

# embeddings go in artifacts/embeddings
EMBED_DIR  = ARTIFACT_DIR / "embeddings"
EMBED_PATH = EMBED_DIR / "embeddings.npz"

# trained models go in artifacts/models
MODEL_DIR      = ARTIFACT_DIR / "models"
PCA_MODEL_PATH = MODEL_DIR / "pca_model.joblib"
SCALER_PATH    = MODEL_DIR / "scaler.joblib"
DBSCAN_PATH    = MODEL_DIR / "dbscan_model.joblib"

# make sure dirs exist at startup
for d in (EMBED_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)
