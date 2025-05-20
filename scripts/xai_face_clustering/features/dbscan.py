import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
import joblib

DBSCAN_SAVE_PATH = "xai_face_clustering/models/dbscan_model.joblib"
SCALER_SAVE_PATH = "xai_face_clustering/models/scaler.joblib"

def apply_dbscan(features, eps=0.5, min_samples=5, fit=True, save=True):
    """
    Fit and apply DBSCAN clustering
    Optionally save and load DBSCAN models using joblib
    
    Standardizes the data before clustering (essential!)
    Saves dbscan_model.joblib and scaler.joblib for consistent transformations later
    We can later load and apply DBSCAN using fit=False   
    """
    if fit:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        clusters = dbscan.fit_predict(features_scaled)

        if save:
            os.makedirs(os.path.dirname(DBSCAN_SAVE_PATH), exist_ok=True)
            joblib.dump(dbscan, DBSCAN_SAVE_PATH)
            joblib.dump(scaler, SCALER_SAVE_PATH)
    else:
        dbscan = joblib.load(DBSCAN_SAVE_PATH)
        scaler = joblib.load(SCALER_SAVE_PATH)
        features_scaled = scaler.transform(features)
        clusters = dbscan.fit_predict(features_scaled)

    return clusters

