# xai_face_clustering/models/clustering/dbscan.py

import os
import joblib
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

DBSCAN_SAVE_PATH  = "artifacts/models/dbscan_model.pkl"
SCALER_SAVE_PATH  = "artifacts/models/scaler.pkl"


def apply_dbscan(features, eps=0.5, min_samples=5, fit=True, save=True):
    """
    Standardize features and cluster with DBSCAN, without parallel progress bar.
    """
    if fit:
        print("🔄 Fitting StandardScaler…")
        scaler = StandardScaler().fit(features)
        feats_scaled = scaler.transform(features)

        print(f"🔄 Running DBSCAN (eps={eps}, min_samples={min_samples}) …")
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, algorithm='auto', n_jobs=1)
        clusters = dbscan.fit_predict(feats_scaled)

        unique = set(clusters)
        n_clusters = len(unique) - (1 if -1 in unique else 0)
        print(f"   → found {n_clusters} clusters (labels={sorted(unique)[:5]} ...)")

        if save:
            os.makedirs(os.path.dirname(DBSCAN_SAVE_PATH), exist_ok=True)
            joblib.dump(dbscan, DBSCAN_SAVE_PATH)
            joblib.dump(scaler, SCALER_SAVE_PATH)
            print("✅ Saved DBSCAN model and scaler.")
    else:
        print("🔄 Loading scaler and DBSCAN model…")
        scaler = joblib.load(SCALER_SAVE_PATH)
        feats_scaled = scaler.transform(features)
        dbscan = joblib.load(DBSCAN_SAVE_PATH)
        clusters = dbscan.fit_predict(feats_scaled)

    return clusters