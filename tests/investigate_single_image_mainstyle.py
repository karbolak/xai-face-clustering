import os
import sys
import numpy as np
import joblib
from PIL import Image
from torchvision import transforms

from scripts.xai_face_clustering.features.cnn_embeddings import extract_embeddings, get_model, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD


MODEL_DIR = "scripts/xai_face_clustering/models"
SVM_MODEL_PATH = "scripts/xai_face_clustering/models/surrogate_model.joblib"

SCALER_PATH = os.path.join(MODEL_DIR, "scaler.joblib")
PCA_PATH = os.path.join(MODEL_DIR, "pca_model.joblib")
EMBED_CACHE = "scripts/xai_face_clustering/features/embeddings.npz"

def print_title(title):
    print("=" * len(title))
    print(title)
    print("=" * len(title))

def main(img_path):
    print_title(f"Investigating image: {img_path}")

    img = Image.open(img_path).convert("RGB")
    print(f"Loaded image size: {img.size}")

    clf = joblib.load(SVM_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    model, _ = get_model("facenet")
    tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    #extracta embedding using main.py function
    #note -> extract_embeddings returns a batch, but we want to retrieve only a single one
    print_title("Extracting embedding")
    embedding, _, _ = extract_embeddings(
        [img],
        filenames=None,
        labels=[0],  # label doesn't matter here
        model_name="facenet",
        cache_path=None  # we do NOT save cache
    )
    emb = embedding[0]
    print("Embedding shape:", emb.shape)
    print("Embedding preview:", emb[:10])

    # scaler
    print_title("Applying StandardScaler")
    emb_scaled = scaler.transform([emb])
    print("Scaled embedding (preview):", emb_scaled[0][:10])

    # PCA
    print_title("Applying PCA")
    emb_pca = pca.transform(emb_scaled)
    print("PCA embedding (preview):", emb_pca[0][:10])

    # predict
    print_title("Classifier Prediction")
    pred_label_num = int(clf.predict(emb_pca)[0])
    pred_label = "Real" if pred_label_num == 0 else "AI"
    print(f"Predicted label: {pred_label} ({pred_label_num})")
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(emb_pca)[0]
        print(f"Prediction probability vector: {proba}")

    #compare with cached embeddings
    if os.path.exists(EMBED_CACHE):
        cached = np.load(EMBED_CACHE, allow_pickle=True)
        cache_embeds = cached["embeddings"]
        cache_labels = cached["labels"]
        # Find the embedding for this file
        img_name = os.path.basename(img_path)
        # If you have a mapping of file order, find and print it

    print_title("DEBUG INFO")
    debug_info = {
        "img_path": img_path,
        "embedding_shape": emb.shape,
        "scaled_embedding_shape": emb_scaled.shape,
        "pca_embedding_shape": emb_pca.shape,
        "pca_embedding_preview": emb_pca[0][:10].tolist(),
        "predicted_label_num": pred_label_num,
        "probability_vector": proba.tolist() if hasattr(clf, "predict_proba") else None
    }
    for k, v in debug_info.items():
        print(f"{k}: {v}")

    print("\n[INFO] Investigation complete.")

if __name__ == "__main__":
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Need to provide image file, in the following format: python -m investigate_single_image_mainstyle.py path/to/image.jpg")
        exit(1)
    main(sys.argv[1])
