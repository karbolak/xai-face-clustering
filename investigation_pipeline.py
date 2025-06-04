import os
import numpy as np
import torch
from PIL import Image
import joblib

from torchvision import transforms

# Adjust paths as needed
SVM_MODEL_PATH = "scripts/xai_face_clustering/models/svm_model.joblib"
PCA_PATH      = "scripts/xai_face_clustering/models/pca_model.joblib"
SCALER_PATH   = "scripts/xai_face_clustering/models/scaler.joblib"
EMBEDDING_MODEL = "facenet"

# using custom embedding extractor:
from scripts.xai_face_clustering.features.cnn_embeddings import get_model, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

def print_title(title):
    print("="*len(title))
    print(title)
    print("="*len(title))

def get_embedding(img: Image.Image, model, tf):
    # Register hook
    activations = {}
    def get_hook(_, __, out):
        activations["feat"] = out.detach()
    for name, module in model.named_modules():
        if name == "last_linear":
            module.register_forward_hook(get_hook)
            break
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        _ = model(x)
    feat = activations["feat"]
    emb = feat.cpu().numpy().squeeze()
    return emb

def investigate_image(img_path):
    print_title(f"Investigating image: {img_path}")
    # Load image
    img = Image.open(img_path).convert("RGB")
    print(f"Loaded image size: {img.size}")

    # Load models/transform
    print_title("Loading models & transforms")
    clf = joblib.load(SVM_MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    model, _ = get_model(EMBEDDING_MODEL)
    tf = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    print("Classifier classes_: ", getattr(clf, "classes_", "None"))
    print("Scaler mean (first 5):", scaler.mean_[:5])
    print("PCA mean (first 5):   ", pca.mean_[:5])
    print()

    # Embedding extraction
    print_title("Extracting embedding")
    emb = get_embedding(img, model, tf)
    print("Embedding shape:", emb.shape)
    print("Embedding preview:", emb[:10])

    # Scaling
    print_title("Applying StandardScaler")
    emb_scaled = scaler.transform([emb])
    print("Scaled embedding (preview):", emb_scaled[0][:10])

    # PCA
    print_title("Applying PCA")
    emb_pca = pca.transform(emb_scaled)
    print("PCA embedding (shape):", emb_pca.shape)
    print("PCA embedding (preview):", emb_pca[0][:10])

    # Prediction
    print_title("Classifier Prediction")
    pred_label_num = int(clf.predict(emb_pca)[0])
    pred_label = "Real" if pred_label_num == 0 else "AI"
    print(f"Predicted label: {pred_label} ({pred_label_num})")

    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(emb_pca)[0]
        print(f"Prediction probability vector: {proba}")
    else:
        print("Classifier has no predict_proba method.")

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

    print("\n\n[INFO] Investigation complete.")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2 or not os.path.isfile(sys.argv[1]):
        print("Usage: python investigate_pipeline.py path/to/image.jpg")
        exit(1)
    investigate_image(sys.argv[1])
