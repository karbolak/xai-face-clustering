import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import joblib
import torch
from torchvision import transforms
import os
import json

from scripts.xai_face_clustering.features.cnn_embeddings import get_model, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

app = FastAPI()

# Load models at startup
SURROGATE_MODEL_PATH = "scripts/xai_face_clustering/models/surrogate_model.joblib"
PCA_PATH = "scripts/xai_face_clustering/models/pca_model.joblib"
CLUSTER_MAP_PATH = "scripts/xai_face_clustering/models/cluster_label_map.json"

surrogate = joblib.load(SURROGATE_MODEL_PATH)
pca = joblib.load(PCA_PATH)
model, layers_to_hook = get_model("facenet")

# Try loading cluster-label mapping
if os.path.exists(CLUSTER_MAP_PATH):
    with open(CLUSTER_MAP_PATH, "r") as f:
        cluster_map = json.load(f)
else:
    cluster_map = None

# Preprocessing
tf = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def get_embedding(img: Image.Image):
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

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Step 1: Read and preprocess
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    emb = get_embedding(image)

    # Step 2: PCA transform
    emb_pca = pca.transform([emb])

    # Step 3: Predict cluster using surrogate
    pred_cluster = surrogate.predict(emb_pca)[0]

    # Step 4: SVM confidence/probability
    if hasattr(surrogate, "predict_proba"):
        proba = surrogate.predict_proba(emb_pca)[0]
    else:
        proba = None

    # Step 5: Cluster→label mapping
    if cluster_map is not None:
        mapped_label_num = cluster_map.get(str(pred_cluster), "Unknown")
        mapped_label = "AI" if mapped_label_num == 1 else "Real" if mapped_label_num == 0 else "Unknown"
    else:
        mapped_label = "AI" if pred_cluster == 1 else "Real"

    # Step 6: Debug output
    debug_info = {
        "predicted_cluster": int(pred_cluster),
        "cluster_map_used": cluster_map,
        "mapped_label": mapped_label,
        "SVM_probability_vector": proba.tolist() if proba is not None else "not available",
        "pca_embedding_preview": emb_pca[0][:10].tolist(),  # show first 10 components
        "pca_embedding_shape": emb_pca.shape
    }

    print("--- API PREDICTION DEBUG ---")
    for k, v in debug_info.items():
        print(f"{k}: {v}")

    # Step 7: Return detailed output for investigation
    return {
        "prediction": mapped_label,
        "debug": debug_info
    }
