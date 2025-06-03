import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import joblib
import torch
from torchvision import transforms
import os
import shap
import matplotlib.pyplot as plt
import base64
from io import BytesIO

from scripts.xai_face_clustering.features.cnn_embeddings import get_model, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

app = FastAPI()

# Load models at startup
SVM_MODEL_PATH = "scripts/xai_face_clustering/models/svm_model.joblib"
PCA_PATH = "scripts/xai_face_clustering/models/pca_model.joblib"
SCALER_PATH = "scripts/xai_face_clustering/models/scaler.joblib"
bg_path = "scripts/xai_face_clustering/models/shap_background.npz"

clf = joblib.load(SVM_MODEL_PATH)
pca = joblib.load(PCA_PATH)
scaler = joblib.load(SCALER_PATH)
background = np.load(bg_path)["background"]
model, layers_to_hook = get_model("facenet")

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
    emb_scaled = scaler.transform([emb])
    emb_pca = pca.transform(emb_scaled)

    # Step 3: Predict label using supervised model
    pred_label_num = int(clf.predict(emb_pca)[0])
    pred_label = "Real" if pred_label_num == 0 else "AI"
    
    # --- SHAP: Build explainer at runtime (never load from file) ---
    # explainer = shap.KernelExplainer(clf.predict_proba, background)
    # shap_vals = explainer.shap_values(emb_pca)
    # shap_img = None
    # try:
    #     fig = plt.figure()
    #     # Robustly handle output shape (list vs ndarray)
    #     if isinstance(shap_vals, list):
    #         shap.plots.waterfall(shap_vals[pred_label_num][0], show=False)
    #     else:
    #         shap.plots.waterfall(shap_vals[0], show=False)
    #     buf = BytesIO()
    #     plt.savefig(buf, format='png')
    #     plt.close(fig)
    #     buf.seek(0)
    #     shap_img = base64.b64encode(buf.read()).decode("utf-8")
    # except Exception as e:
    #     print(f"Could not create SHAP plot: {e}")

    # Step 4: SVM confidence/probability
    if hasattr(clf, "predict_proba"):
        proba = clf.predict_proba(emb_pca)[0].tolist()
    else:
        proba = None

    debug_info = {
        "predicted_label_num": pred_label_num,
        "probability_vector": proba,
        "pca_embedding_preview": emb_pca[0][:10].tolist(),
        "pca_embedding_shape": emb_pca.shape
    }
    print("--- API PREDICTION DEBUG ---")
    for k, v in debug_info.items():
        print(f"{k}: {v}")

    return {
        "prediction": pred_label,
        "debug": debug_info,
        # "shap_plot": shap_img
    }
