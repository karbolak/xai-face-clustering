import argparse
import os
import joblib
import numpy as np
import shap
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from scripts.xai_face_clustering.data.loader import load_images
from scripts.xai_face_clustering.features.cnn_embeddings import extract_embeddings
from scripts.xai_face_clustering.features.pca import apply_pca
from scripts.xai_face_clustering.features.pca_variance_plot import plot_pca_variance

# ── Paths for artifacts ────────────────────────────────────────────────
MODEL_DIR              = "scripts/xai_face_clustering/models"
EMBED_CACHE            = "scripts/xai_face_clustering/features/embeddings.npz"
PCA_VARIANCE_PLOT_PATH = "scripts/xai_face_clustering/features/exploratory_plots/pca_explained_variance.png"
SCALER_PATH            = os.path.join(MODEL_DIR, "scaler.joblib")
PCA_PATH               = os.path.join(MODEL_DIR, "pca_model.joblib")
SVM_MODEL_PATH         = os.path.join(MODEL_DIR, "svm_model.joblib")
LOGREG_MODEL_PATH      = os.path.join(MODEL_DIR, "logreg_model.joblib")

def main(args):
    # Ensure directories exist
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(os.path.dirname(PCA_VARIANCE_PLOT_PATH), exist_ok=True)

    # 1. Load or cache embeddings
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

    # 2. Train/test split
    print("[INFO] Splitting train/test set…")
    X_train, X_test, y_train, y_test = train_test_split(
        embeddings,
        labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )

    # 3. PCA explained variance plot
    print("[INFO] Plotting PCA explained variance…")
    plot_pca_variance(
        X_train,
        PCA_VARIANCE_PLOT_PATH
    )

    # 4. Scale + PCA
    print(f"[INFO] Applying PCA ({args.pca_components} components)…")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled  = scaler.transform(X_test)
    joblib.dump(scaler, SCALER_PATH)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=args.pca_components)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca  = pca.transform(X_test_scaled)
    joblib.dump(pca, PCA_PATH)

    # 5. Supervised classifiers
    print("[INFO] Training Logistic Regression…")
    logreg = LogisticRegression(max_iter=1000)
    logreg.fit(X_train_pca, y_train)
    y_pred_logreg = logreg.predict(X_test_pca)
    print("\n[Logistic Regression] Results on test set:")
    print(confusion_matrix(y_test, y_pred_logreg))
    print(classification_report(y_test, y_pred_logreg, target_names=['Real', 'AI']))
    joblib.dump(logreg, LOGREG_MODEL_PATH)
    print(f"[INFO] Saved Logistic Regression model to {LOGREG_MODEL_PATH}")

    print("[INFO] Training SVM (RBF kernel)…")
    svm = SVC(kernel="rbf", probability=True)
    svm.fit(X_train_pca, y_train)
    y_pred_svm = svm.predict(X_test_pca)
    print("\n[SVM] Results on test set:")
    print(confusion_matrix(y_test, y_pred_svm))
    print(classification_report(y_test, y_pred_svm, target_names=['Real', 'AI']))
    joblib.dump(svm, SVM_MODEL_PATH)
    print(f"[INFO] Saved SVM model to {SVM_MODEL_PATH}")
    
    print("classes_:", svm.classes_)
    print("Scaler mean:", scaler.mean_[:5])
    print("PCA mean:", pca.mean_[:5])
    print("First test emb_pca:", X_test_pca[0][:5])
    print("Pred proba on first test:", svm.predict_proba(X_test_pca[:1]))

    # --- SHAP explanation for SVM --- shap.force_plot or shap.bar_plot for other vis styles
    print("[INFO] Fitting SHAP explainer for SVM...")
    background = X_train_pca[:10]
    np.savez(os.path.join(MODEL_DIR, "shap_background.npz"), background=background)

    explainer = shap.KernelExplainer(svm.predict_proba, background)
    shap_values = explainer.shap_values(X_test_pca[:5])  # Only a few samples

    print(f"shap_values type: {type(shap_values)}, shape/len: "
        f"{shap_values.shape if hasattr(shap_values, 'shape') else len(shap_values)}")

    # Save SHAP values for documentation
    np.savez(os.path.join(MODEL_DIR, "shap_test_values.npz"),
            shap_values=shap_values, X_test_pca=X_test_pca[:5], y_test=y_test[:5])

    # Plotting (robust to both list and ndarray)
    plt.figure(figsize=(12, 5))
    if isinstance(shap_values, list):
        if len(shap_values) == 2:
            shap.summary_plot(shap_values[1], X_test_pca[:5], show=False)  # For class 1 (AI)
        else:
            shap.summary_plot(shap_values[0], X_test_pca[:5], show=False)
    else:
        shap.summary_plot(shap_values, X_test_pca[:5], show=False)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, "shap_summary_plot.png"))
    plt.close()
    print("[INFO] SHAP summary plot generated.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Supervised Face Classification Pipeline")
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
        default=100,
        help="Number of PCA components to retain"
    )
    args = parser.parse_args()
    main(args)
