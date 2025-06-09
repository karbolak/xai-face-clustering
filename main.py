import argparse
import os
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.cluster import SpectralClustering
from sklearn.linear_model import LogisticRegression

from scripts.xai_face_clustering.data.loader import load_images
from scripts.xai_face_clustering.features.cnn_embeddings import extract_embeddings
from scripts.xai_face_clustering.features.pca import apply_pca
from scripts.xai_face_clustering.features.pca_variance_plot import plot_pca_variance
from scripts.xai_face_clustering.features.dbscan_plot import plot_dbscan_kdistance
from scripts.xai_face_clustering.models.clustering import cluster_embeddings
from scripts.xai_face_clustering.models.surrogate import train_surrogate_model
from scripts.xai_face_clustering.models.xai import run_shap_explanation

MODEL_DIR = "scripts/xai_face_clustering/models"
EMBED_CACHE = "scripts/xai_face_clustering/features/embeddings.npz"
PCA_VARIANCE_PLOT_PATH = "scripts/xai_face_clustering/features/exploratory_plots/pca_explained_variance.png"
DBSCAN_KDIST_PLOT_PATH = "scripts/xai_face_clustering/features/exploratory_plots/dbscan_kdistance.png"
SURROGATE_MODEL_PATH = os.path.join(MODEL_DIR, "surrogate_model.joblib")
CLUSTER_MAP_PATH = os.path.join(MODEL_DIR, "cluster_label_map.json")


def main(args):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PCA_VARIANCE_PLOT_PATH), exist_ok=True)

    if os.path.exists(EMBED_CACHE):
        print("[INFO] Loading cached embeddings...")
        data = np.load(EMBED_CACHE, allow_pickle=True)
        embeddings = data["embeddings"]
        labels = data["labels"].tolist()
    else:
        print(f"[INFO] Loading images from {args.data_dir}...")
        images, labels, _ = load_images(args.data_dir, as_numpy_list=True)
        print(f"[INFO] Extracting embeddings for {len(images)} images...")
        embeddings, _, labels = extract_embeddings(
            images, filenames=None, labels=labels, model_name=args.model, cache_path=EMBED_CACHE
        )
        print("[INFO] Embedding extraction complete.")

    print("[INFO] Splitting data into training and testing sets...")
    X_train, X_test, y_train_orig, y_test_orig = train_test_split(
        embeddings, labels, test_size=0.2, random_state=42, stratify=labels
    )

    print("[INFO] Plotting PCA variance...")
    plot_pca_variance(X_train, PCA_VARIANCE_PLOT_PATH)

    print(f"[INFO] Reducing dimensions to {args.pca_components} with PCA...")
    X_train_pca = apply_pca(X_train, args.pca_components, fit=True, save=True)
    X_test_pca = apply_pca(X_test, args.pca_components, fit=False, save=False)

    if args.cluster_method == "dbscan" and args.plot_kdist:
        print("[INFO] Creating DBSCAN k-distance plot...")
        plot_dbscan_kdistance(X_train_pca, DBSCAN_KDIST_PLOT_PATH, k=args.kdist_k)

    print(f"[INFO] Running {args.cluster_method} clustering on training data...")
    y_train_cluster = cluster_embeddings(X_train_pca, method=args.cluster_method, evaluate_stability=True,
                                         true_labels=y_train_orig, eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)

    print("[INFO] Running clustering on test data...")
    y_test_cluster = cluster_embeddings(X_test_pca, method=args.cluster_method, true_labels=y_test_orig,
                                        eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)

    np.savez("scripts/xai_face_clustering/features/test_clusters.npz", y_test_cluster=y_test_cluster, y_test_orig=y_test_orig)

    print("[INFO] Training surrogate model...")
    surrogate = train_surrogate_model(X_train_pca, y_train_cluster, X_test_pca, y_test_cluster, method=args.surrogate)
    joblib.dump(surrogate, SURROGATE_MODEL_PATH)

    print("[INFO] Mapping clusters to real/AI labels...")
    cluster_map = {}
    unique_clusters = np.unique(y_train_cluster)
    for cluster_id in unique_clusters:
        indices = [i for i, val in enumerate(y_train_cluster) if val == cluster_id]
        corresponding_labels = [y_train_orig[i] for i in indices]
        most_common_label = max(set(corresponding_labels), key=corresponding_labels.count)
        cluster_map[int(cluster_id)] = int(most_common_label)

    with open(CLUSTER_MAP_PATH, "w") as f:
        json.dump(cluster_map, f)

    y_pred_clusters = surrogate.predict(X_test_pca)
    y_pred_final = [cluster_map.get(int(c), -1) for c in y_pred_clusters]

    print("\n=== Final Model Results ===")
    #confusion matrix with false positives, false negatives, true negatives 
    print(confusion_matrix(y_test_orig, y_pred_final))
    print(classification_report(y_test_orig, y_pred_final, target_names=["Real", "AI"]))

    if -1 in y_pred_final:
        print(f"[WARNING] {y_pred_final.count(-1)} test predictions could not be mapped.")

    print("\n=== Baseline: Spectral Clustering + Logistic Regression ===")
    n_clusters = len(np.unique(y_train_orig))
    spectral = SpectralClustering(n_clusters=n_clusters, affinity='nearest_neighbors', random_state=42, n_neighbors=10)
    y_train_spectral = spectral.fit_predict(X_train_pca)

    cluster_centers = []
    for cluster_id in range(n_clusters):
        points = X_train_pca[y_train_spectral == cluster_id]
        if len(points) > 0:
            center = np.mean(points, axis=0)
        else:
            center = np.zeros(X_train_pca.shape[1])
        cluster_centers.append(center)

    distances = cdist(X_test_pca, np.vstack(cluster_centers))
    y_test_spectral = np.argmin(distances, axis=1)

    spectral_cluster_map = {}
    for cluster_id in np.unique(y_train_spectral):
        idxs = np.where(y_train_spectral == cluster_id)[0]
        labels_for_cluster = [y_train_orig[i] for i in idxs]
        if labels_for_cluster:
            spectral_cluster_map[int(cluster_id)] = max(set(labels_for_cluster), key=labels_for_cluster.count)
        else:
            spectral_cluster_map[int(cluster_id)] = -1

    baseline_logreg = LogisticRegression(max_iter=1000)
    baseline_logreg.fit(X_train_pca, y_train_spectral)
    y_pred_spectral = baseline_logreg.predict(X_test_pca)
    y_pred_final_spectral = [spectral_cluster_map.get(int(c), -1) for c in y_pred_spectral]

    print(confusion_matrix(y_test_orig, y_pred_final_spectral))
    print(classification_report(y_test_orig, y_pred_final_spectral, target_names=["Real", "AI"]))

    if -1 in y_pred_final_spectral:
        print(f"[WARNING] {y_pred_final_spectral.count(-1)} test samples not mapped.")

    if args.shap:
        print("[INFO] Running SHAP explanations...")
        run_shap_explanation(surrogate, X_test_pca, y_test_cluster)
        print("[INFO] SHAP explanations complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAI Face Clustering Pipeline")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="scripts/xai_face_clustering/data/Human_Faces_Dataset",
        help="Path to your face dataset"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="facenet",
        help="Pretrained CNN model for embeddings"
    )
    parser.add_argument(
        "--pca_components",
        type=int,
        default=475, # 95%
        help="Number of PCA components to retain"
    )
    parser.add_argument(
        "--cluster_method",
        type=str,
        default="gmm",
        choices=["kmeans", "dbscan", "gmm"],
        help="Clustering algorithm to use"
    )
    parser.add_argument(
        "--surrogate",
        type=str,
        default="svm",
        choices=["logreg", "tree", "svm"],
        help="Surrogate classifier type"
    )
    parser.add_argument(
        "--dbscan_eps",
        type=float,
        default=70,
        help="DBSCAN eps (neighborhood radius)"
    )
    parser.add_argument(
        "--dbscan_min_samples",
        type=int,
        default=5,
        help="DBSCAN min_samples (min points in neighborhood)"
    )
    parser.add_argument(
        "--plot_kdist",
        action="store_true",
        help="Whether to plot the k-distance graph for DBSCAN ε selection"
    )
    parser.add_argument(
        "--kdist_k",
        type=int,
        default=5,
        help="Which neighbor (k) to use for the k-distance plot"
    )
    parser.add_argument(
        "--shap",
        action="store_true",
        help="Whether to run SHAP explanations"
    )

    args = parser.parse_args()
    main(args)
