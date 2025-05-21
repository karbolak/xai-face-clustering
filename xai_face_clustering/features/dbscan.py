import os
import joblib
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

DBSCAN_SAVE_PATH = "artifacts/models/dbscan_model.joblib"
SCALER_SAVE_PATH = "artifacts/models/scaler.joblib"

def apply_dbscan(features, eps=0.5, min_samples=5, fit=True, save=True):
    """
    Standardize features, cluster with DBSCAN, and optionally save/load the model & scaler.
    """
    if fit:
        scaler = StandardScaler().fit(features)
        feats_scaled = scaler.transform(features)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(feats_scaled)

        if save:
            os.makedirs(os.path.dirname(DBSCAN_SAVE_PATH), exist_ok=True)
            joblib.dump(dbscan, DBSCAN_SAVE_PATH)
            joblib.dump(scaler, SCALER_SAVE_PATH)
    else:
        scaler = joblib.load(SCALER_SAVE_PATH)
        feats_scaled = scaler.transform(features)
        dbscan = joblib.load(DBSCAN_SAVE_PATH)
        clusters = dbscan.fit_predict(feats_scaled)

    return clusters
