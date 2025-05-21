# xai_face_clustering/features/dbscan.py

import os
import pickle
import warnings
import numpy as np
from time import time

import hdbscan                  # conda install -c conda-forge hdbscan
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestCentroid

# Where we stash model & scaler
HDBSCAN_SAVE_PATH = "artifacts/models/hdbscan_model.pkl"
SCALER_SAVE_PATH  = "artifacts/models/scaler.pkl"

# Quiet that sklearn rename warning
warnings.filterwarnings(
    "ignore",
    message="'force_all_finite' was renamed to 'ensure_all_finite'"
)

def apply_dbscan(
    features: np.ndarray,
    eps: float = 0.0,
    min_samples: int = 5,
    fit: bool = True,
    save: bool = True,
    subsample_threshold: int = 10000,
    subsample_size: int = 10000
) -> np.ndarray:
    """
    HDBSCAN with:
     - PCA'd / scaled features in
     - subsampling for speed if N > subsample_threshold
     - NearestCentroid to assign the remainder
     - timing & progress prints
    """

    if fit:
        N = features.shape[0]

        # 1️⃣  Scale
        print("1️⃣  Fitting StandardScaler…")
        t0 = time()
        scaler = StandardScaler().fit(features)
        Xs = scaler.transform(features)
        print(f"   → scaler done in {time()-t0:.2f}s")

        # Decide whether to subsample
        if N > subsample_threshold:
            print(f"2️⃣  Dataset size {N} > {subsample_threshold}, subsampling {subsample_size}")
            # choose random subset
            idx = np.random.choice(N, subsample_size, replace=False)
            X_sub = Xs[idx]
            t1 = time()
            print("   • Running HDBSCAN on subsample…")
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_samples,
                cluster_selection_epsilon=eps,
                metric="euclidean",
                core_dist_n_jobs=-1,
                gen_min_span_tree=False,
                prediction_data=True
            )
            labels_sub = clusterer.fit_predict(X_sub)
            dur1 = time() - t1
            n_sub = len(set(labels_sub)) - (1 if -1 in labels_sub else 0)
            print(f"   → subsample done in {dur1:.2f}s; {n_sub} clusters")

            # assign the rest
            print("3️⃣  Assigning remaining points via NearestCentroid…")
            t2 = time()
            nc = NearestCentroid().fit(X_sub, labels_sub)
            print("   • Fitted, now predicting on full set", flush=True)

            # Sanity-check dtype
            print(f"   • Xs type: {type(Xs)}, dtype: {Xs.dtype}", flush=True)

            labels = nc.predict(Xs)

            dur2 = time() - t2
            print(f"   → assignment done in {dur2:.3f}s", flush=True)

            # *skip caching* since we fit on a subsample
            if save:
                print("⚠️  Skipping cache save when subsampling")
        else:
            # full‐data HDBSCAN
            print(f"2️⃣  Running HDBSCAN on full dataset ({N} points)…")
            t1 = time()
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_samples,
                cluster_selection_epsilon=eps,
                metric="euclidean",
                core_dist_n_jobs=-1,
                gen_min_span_tree=False,
                prediction_data=True
            )
            labels = clusterer.fit_predict(Xs)
            dur1 = time() - t1
            ncl = len(set(labels)) - (1 if -1 in labels else 0)
            print(f"   → HDBSCAN done in {dur1:.2f}s; found {ncl} clusters")

            # 3️⃣  Cache scaler & model
            if save:
                print("3️⃣  Saving scaler & HDBSCAN model…")
                os.makedirs(os.path.dirname(SCALER_SAVE_PATH), exist_ok=True)
                with open(SCALER_SAVE_PATH, "wb") as f:
                    pickle.dump(scaler, f)
                with open(HDBSCAN_SAVE_PATH, "wb") as f:
                    pickle.dump(clusterer, f)
                print("   → cache saved.")

    else:
        # reload & predict
        print("🔄 Loading scaler & HDBSCAN model…")
        with open(SCALER_SAVE_PATH, "rb") as f:
            scaler = pickle.load(f)
        with open(HDBSCAN_SAVE_PATH, "rb") as f:
            clusterer = pickle.load(f)

        Xs = scaler.transform(features)
        print("🔄 Predicting on full dataset…")
        labels, _ = hdbscan.approximate_predict(clusterer, Xs)
        ncl = len(set(labels)) - (1 if -1 in labels else 0)
        print(f"   → prediction done; {ncl} clusters")

    return labels
