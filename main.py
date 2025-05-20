import argparse
import os
import numpy as np
from sklearn.model_selection import train_test_split
from xai_face_clustering.data.loader import load_images
from xai_face_clustering.features.cnn_embeddings import extract_embeddings
from xai_face_clustering.features.pca import apply_pca
from xai_face_clustering.models.clustering import cluster_embeddings
from xai_face_clustering.models.surrogate import train_surrogate_model
from xai_face_clustering.models.xai import run_shap_explanation
from xai_face_clustering.features.pca_variance_plot import plot_pca_variance

def main(args):
    cache_path = "xai_face_clustering/features/embeddings.npz"

    if os.path.exists(cache_path):
        print("[INFO] Cached embeddings found, skipping image loading...")
        images = None
        filenames = None
        labels = None
    else:
        print("[INFO] Loading images...")
        images, labels, filenames = load_images(args.data_dir)

    print("[INFO] Extracting embeddings...")
    embeddings, filenames, labels = extract_embeddings(
        images, filenames=filenames, labels=labels, model_name=args.model, cache_path=cache_path
    )

    print("[INFO] Splitting train/test set...")
    X_train, X_test = train_test_split(embeddings, test_size=0.2, random_state=42)

    plot_pca_variance(X_train, "xai_face_clustering/features/exploratory_plots/pca_explained_variance.png")

    print("[INFO] Applying PCA...")
    X_train_pca = apply_pca(X_train, n_components=args.pca_components, fit=True)
    X_test_pca = apply_pca(X_test, n_components=args.pca_components, fit=False)
    
    # X_train_pca = X_train
    # X_test_pca = X_test

    print("[INFO] Clustering train/test sets...")
    y_train = cluster_embeddings(X_train_pca, method=args.cluster_method, evaluate_stability=True)
    y_test = cluster_embeddings(X_test_pca, method=args.cluster_method)

    print("[INFO] Training surrogate model...")
    surrogate_model = train_surrogate_model(X_train_pca, y_train, X_test_pca, y_test, method=args.surrogate)

    print("[INFO] Running SHAP explanation...")
    run_shap_explanation(surrogate_model, X_test_pca, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAI Face Clustering Pipeline")
    parser.add_argument("--data_dir", type=str, default="xai_face_clustering/data/Human_Faces_Dataset", help="Path to dataset")
    parser.add_argument("--model", type=str, default="facenet", help="Pretrained CNN model name")
    parser.add_argument("--pca_components", type=int, default=100, help="PCA component count")
    parser.add_argument("--cluster_method", type=str, default="kmeans", choices=["kmeans", "dbscan"], help="Clustering method")
    parser.add_argument("--surrogate", type=str, default="svm", choices=["logreg", "tree", "svm"], help="Surrogate classifier")

    args = parser.parse_args()
    main(args)
