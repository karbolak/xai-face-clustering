import argparse
from scripts.xai_face_clustering.data.loader import load_images
from scripts.xai_face_clustering.features.cnn_embeddings import extract_embeddings
from scripts.xai_face_clustering.features.pca import apply_pca
from scripts.xai_face_clustering.models.clustering import cluster_embeddings
from scripts.xai_face_clustering.models.surrogate import train_surrogate_model
from scripts.xai_face_clustering.models.xai import run_shap_explanation


def main(args):
    """_summary_
        This script will serve as our entry point and support modular execution
        (e.g., clustering only, or visualisation only).

        Args:
            args (_type_): _description_
    """
    print("[INFO] Loading images...")
    images, labels, filenames = load_images(args.data_dir)

    print("[INFO] Extracting embeddings...")
    embeddings = extract_embeddings(images, model_name=args.model)

    print("[INFO] Applying PCA...")
    reduced = apply_pca(embeddings, n_components=args.pca_components)

    print("[INFO] Clustering embeddings...")
    cluster_ids = cluster_embeddings(reduced, method=args.cluster_method)

    print("[INFO] Training surrogate model...")
    surrogate_model = train_surrogate_model(reduced, cluster_ids, method=args.surrogate)

    print("[INFO] Running SHAP explanation...")
    run_shap_explanation(surrogate_model, reduced, cluster_ids)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XAI Face Clustering Pipeline")
    parser.add_argument("--data_dir", type=str, default="xai_face_clustering/data/Human_Faces_ds", help="Path to dataset")
    parser.add_argument("--model", type=str, default="resnet50", help="Pretrained CNN model name")
    parser.add_argument("--pca_components", type=int, default=100, help="PCA component count")
    parser.add_argument("--cluster_method", type=str, default="kmeans", choices=["kmeans", "dbscan"], help="Clustering method")
    parser.add_argument("--surrogate", type=str, default="logreg", choices=["logreg", "tree"], help="Surrogate classifier")

    args = parser.parse_args()
    main(args)
