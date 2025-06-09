import io
import numpy as np
from fastapi import FastAPI, File, UploadFile, Query
from fastapi.responses import Response
from PIL import Image
import joblib
import torch
from torchvision import transforms
import os
import json
import shap
import matplotlib.pyplot as plt
from io import BytesIO

from scripts.xai_face_clustering.features.cnn_embeddings import get_model, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

app = FastAPI()

SURROGATE_MODEL_PATH = "scripts/xai_face_clustering/models/surrogate_model.joblib"
PCA_PATH = "scripts/xai_face_clustering/models/pca_model.joblib"
CLUSTER_MAP_PATH = "scripts/xai_face_clustering/models/cluster_label_map.json"
SHAP_BG_PATH = "scripts/xai_face_clustering/models/shap_background.npz"

surrogate = joblib.load(SURROGATE_MODEL_PATH)
pca = joblib.load(PCA_PATH)
model, layers_to_hook = get_model("facenet")

if os.path.exists(CLUSTER_MAP_PATH):
    with open(CLUSTER_MAP_PATH, "r") as f:
        cluster_map = json.load(f)
else:
    cluster_map = None

if os.path.exists(SHAP_BG_PATH):
    background = np.load(SHAP_BG_PATH)["background"]
else:
    background = None

tf = transforms.Compose([
    transforms.Resize(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

def get_embedding(img: Image.Image):
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
async def predict(
    debug: str = Query("false", description="Return debug info for this image"),
    file: UploadFile = File(...),
):
    debug = str(debug).strip().lower() in ("true", "1", "yes")
    
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    emb = get_embedding(image)

    emb_pca = pca.transform([emb])

    pred_cluster = surrogate.predict(emb_pca)[0]

    if hasattr(surrogate, "predict_proba"):
        proba = surrogate.predict_proba(emb_pca)[0]
    else:
        proba = None

    if cluster_map is not None:
        mapped_label_num = cluster_map.get(str(pred_cluster), "Unknown")
        mapped_label = "AI" if mapped_label_num == 1 else "Real" if mapped_label_num == 0 else "Unknown"
    else:
        mapped_label = "AI" if pred_cluster == 1 else "Real"

    response = {
        "prediction": mapped_label,
    }
    if debug:
        debug_info = {
            "predicted_cluster": int(pred_cluster),
            "cluster_map_used": cluster_map,
            "mapped_label": mapped_label,
            "SVM_probability_vector": proba.tolist() if proba is not None else "not available",
            "pca_embedding_preview": emb_pca[0][:10].tolist(),
            "pca_embedding_shape": emb_pca.shape
        }
        response["debug"] = debug_info

    return response

@app.post("/shap_image", tags=["Visualization"])
async def shap_image(
    file: UploadFile = File(...),
):
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    emb = get_embedding(image)
    emb_pca = pca.transform([emb])

    if background is None:
        return Response(content="No SHAP background available", media_type="text/plain")
    try:
        explainer = shap.KernelExplainer(surrogate.predict_proba, background)
        shap_vals = explainer.shap_values(emb_pca)
        fig = plt.figure()
        shap.plots.waterfall(shap_vals[0], show=False)
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return Response(content=buf.read(), media_type="image/png")
    except Exception as e:
        return Response(content=f"Error generating SHAP plot: {e}", media_type="text/plain")
