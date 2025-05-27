import argparse
import os
import json
import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from scripts.xai_face_clustering.data.loader import load_images
from scripts.xai_face_clustering.features.cnn_embeddings import extract_embeddings
from scripts.xai_face_clustering.features.pca import apply_pca
from scripts.xai_face_clustering.features.pca_variance_plot import plot_pca_variance
from scripts.xai_face_clustering.features.dbscan_plot import plot_dbscan_kdistance
from scripts.xai_face_clustering.models.clustering import cluster_embeddings
from scripts.xai_face_clustering.models.surrogate import train_surrogate_model
from scripts.xai_face_clustering.models.xai import run_shap_explanation

# ── Paths for artifacts ────────────────────────────────────────────────
MODEL_DIR                   = "scripts/xai_face_clustering/models"
EMBED_CACHE                 = "scripts/xai_face_clustering/features/embeddings.npz"
PCA_VARIANCE_PLOT_PATH      = "scripts/xai_face_clustering/features/exploratory_plots/pca_explained_variance.png"
DBSCAN_KDIST_PLOT_PATH      = "scripts/xai_face_clustering/features/exploratory_plots/dbscan_kdistance.png"
SURROGATE_MODEL_PATH        = os.path.join(MODEL_DIR, "surrogate_model.joblib")
CLUSTER_MAP_PATH            = os.path.join(MODEL_DIR, "cluster_label_map.json")

def main(args):
    # Ensure directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PCA_VARIANCE_PLOT_PATH), exist_ok=True)

    # ── 1) Load or cache embeddings ────────────────────────────────────────
    if os.path.exists(EMBED_CACHE):
        print("[INFO] Cached embeddings found; loading…")
        data = np.load(EMBED_CACHE, allow_pickle=True)
        embeddings = data["embeddings"]
        labels     = data["labels"].tolist()
    else:
        print(f"[INFO] Loading images from {args.data_dir}…")
        images, labels, _ = load_images(args.data_dir, as_numpy_list=True)
        print(f"[INFO] Loaded {len(images)} images. Starting embedding extraction...")
        embeddings, _, labels = extract_embeddings(
            images,
            filenames=None,
            labels=labels,
            model_name=args.model,
            cache_path=EMBED_CACHE
        )
        print(f"[INFO] Embedding extraction complete.")

    # ── 2) Train/test split ────────────────────────────────────────────────
    print("[INFO] Splitting train/test set…")
    X_train, X_test, y_train_orig, y_test_orig = train_test_split(
        embeddings,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # ── 3) PCA explained variance plot ────────────────────────────────────
    print("[INFO] Plotting PCA explained variance…")
    plot_pca_variance(
        X_train,
        PCA_VARIANCE_PLOT_PATH
    )

    # ── 4) Scale + PCA ────────────────────────────────────────────────────
    print(f"[INFO] Applying PCA ({args.pca_components} components)…")
    X_train_pca = apply_pca(
        X_train,
        n_components=args.pca_components,
        fit=True,
        save=True
    )
    X_test_pca = apply_pca(
        X_test,
        n_components=args.pca_components,
        fit=False,
        save=False
    )

    # ── 4.5) Optional k-distance plot for DBSCAN ε selection ───────────────
    if args.cluster_method == "dbscan" and args.plot_kdist:
        print("[INFO] Generating k-distance plot for DBSCAN…")
        plot_dbscan_kdistance(
            X_train_pca,
            DBSCAN_KDIST_PLOT_PATH,
            k=args.kdist_k
        )

    # ── 5) Clustering ──────────────────────────────────────────────────────
    print(f"[INFO] Clustering training set ({args.cluster_method})…")
    y_train_cluster = cluster_embeddings(
        X_train_pca,
        method=args.cluster_method,
        evaluate_stability=True,
        true_labels=y_train_orig,
        eps=args.dbscan_eps,
        min_samples=args.dbscan_min_samples
    )

    print(f"[INFO] Clustering test set ({args.cluster_method})…")
    y_test_cluster = cluster_embeddings(
        X_test_pca,
        method=args.cluster_method,
        true_labels=y_test_orig,
        eps=args.dbscan_eps,
        min_samples=args.dbscan_min_samples
    )

    # ── 6) Train surrogate classifier ──────────────────────────────────────
    print(f"[INFO] Training surrogate ({args.surrogate})…")
    surrogate = train_surrogate_model(
        X_train_pca, y_train_cluster,
        X_test_pca,  y_test_cluster,
        method=args.surrogate
    )
    joblib.dump(surrogate, SURROGATE_MODEL_PATH)
    print(f"[INFO] Saved surrogate model to {SURROGATE_MODEL_PATH}")

    # ── 7) Build & save cluster→real/fake map ──────────────────────────────
    print("[INFO] Building cluster→real/fake map…")
    cluster_map = {}
    for cid in np.unique(y_train_cluster):
        idxs = np.where(y_train_cluster == cid)[0]
        orig = [y_train_orig[i] for i in idxs]
        cluster_map[int(cid)] = int(np.bincount(orig).argmax())
    with open(CLUSTER_MAP_PATH, "w") as f:
        json.dump(cluster_map, f)
    print(f"[INFO] Saved cluster→label map to {CLUSTER_MAP_PATH}")

    # ── 8) Optional SHAP explanations ───────────────────────────────────────
    if args.shap:
        print("[INFO] Generating SHAP explanations…")
        run_shap_explanation(surrogate, X_test_pca, y_test_cluster)
        print("[INFO] SHAP explanations done.")

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
        default=50,
        help="Number of PCA components to retain"
    )
    parser.add_argument(
        "--cluster_method",
        type=str,
        default="kmeans",
        choices=["kmeans", "dbscan"],
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
