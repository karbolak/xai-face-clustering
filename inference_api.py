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
import base64
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
    shap: str = Query("false", description="Return SHAP explanation for this image"),
    debug: str = Query("false", description="Return debug info for this image"),
    file: UploadFile = File(...),
):
    # Parse query params as booleans (robust to curl/python)
    print(f"RAW debug param: {debug!r}")
    shap = str(shap).strip().lower() in ("true", "1", "yes")
    debug = str(debug).strip().lower() in ("true", "1", "yes")
    print("PARSED debug value:", debug)
    
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

    # Step 6: SHAP plot (only if requested)
    shap_img = None
    if shap and background is not None:
        try:
            explainer = shap.KernelExplainer(surrogate.predict_proba, background)
            shap_vals = explainer.shap_values(emb_pca)
            fig = plt.figure()
            if isinstance(shap_vals, list):
                shap.plots.waterfall(shap_vals[int(mapped_label_num)][0], show=False)
            else:
                shap.plots.waterfall(shap_vals[0], show=False)
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            buf.seek(0)
            shap_img = base64.b64encode(buf.read()).decode("utf-8")
        except Exception as e:
            shap_img = None

    # Step 7: Debug output
    debug_info = None
    if debug:
        debug_info = {
            "debug": debug,
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

    # Step 8: Return output
    response = {
        "prediction": mapped_label,
    }
    if debug:
        response["debug"] = debug_info
    if shap and shap_img is not None:
        response["shap_plot"] = shap_img

    print("FINAL RESPONSE:", response)
    return response


@app.post("/shap_image", tags=["Visualization"])
async def shap_image(
    file: UploadFile = File(...),
):
    # Step 1: Read and preprocess
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    emb = get_embedding(image)
    emb_pca = pca.transform([emb])
    pred_cluster = surrogate.predict(emb_pca)[0]

    if background is None:
        return Response(content="No SHAP background available", media_type="text/plain")
    try:
        # Get the mapped label index (0 or 1)
        if cluster_map is not None:
            mapped_label_num = cluster_map.get(str(pred_cluster), None)
            if mapped_label_num not in (0, 1):
                mapped_label_num = 0  # fallback (shouldn't happen if map is correct)
        else:
            mapped_label_num = int(pred_cluster)

        # SHAP explanation
        explainer = shap.KernelExplainer(surrogate.predict_proba, background)
        shap_vals = explainer.shap_values(emb_pca)

        fig = plt.figure()
        # If shap_vals is a list (multiclass), select correct class by mapped_label_num
        if isinstance(shap_vals, list) and len(shap_vals) == 2:
            shap.plots.waterfall(shap_vals[mapped_label_num][0], show=False)
        else:
            shap.plots.waterfall(shap_vals[0], show=False)

        plt.title(f"SHAP Waterfall: {'AI' if mapped_label_num == 1 else 'Real'}")
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        buf.seek(0)
        return Response(content=buf.read(), media_type="image/png")
    except Exception as e:
        return Response(content=f"Error generating SHAP plot: {e}", media_type="text/plain")
