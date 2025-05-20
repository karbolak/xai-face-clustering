import os
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

PCA_SAVE_PATH = "scripts/xai-face-clustering/models/pca_model.joblib"
SCALER_SAVE_PATH = "scripts/xai-face-clustering/models/scaler.joblib"

def apply_pca(features, n_components=100, fit=True, save=True):
    """
        Fit and apply PCA
        Optionally save and load PCA models using joblib
        
        Standardizes the data before PCA (essential!)
        Saves pca_model.joblib and scaler.joblib for consistent transformations later
        We can later load and apply PCA using fit=False   
    """
    if fit:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        pca = PCA(n_components=n_components)
        reduced = pca.fit_transform(features_scaled)

        if save:
            os.makedirs(os.path.dirname(PCA_SAVE_PATH), exist_ok=True)
            joblib.dump(pca, PCA_SAVE_PATH)
            joblib.dump(scaler, SCALER_SAVE_PATH)
    else:
        pca = joblib.load(PCA_SAVE_PATH)
        scaler = joblib.load(SCALER_SAVE_PATH)
        features_scaled = scaler.transform(features)
        reduced = pca.transform(features_scaled)

    return reduced
