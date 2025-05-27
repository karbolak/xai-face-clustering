import io
import numpy as np
from fastapi import FastAPI, File, UploadFile
from PIL import Image
import joblib
import torch
from torchvision import transforms

from scripts.xai_face_clustering.features.cnn_embeddings import extract_embeddings, get_model, IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD

app = FastAPI()

# Load models at startup
SURROGATE_MODEL_PATH = "scripts/xai_face_clustering/models/surrogate_model.joblib"
PCA_PATH = "scripts/xai_face_clustering/models/pca_model.joblib"

surrogate = joblib.load(SURROGATE_MODEL_PATH)
pca = joblib.load(PCA_PATH)
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
    image = Image.open(io.BytesIO(await file.read())).convert("RGB")
    emb = get_embedding(image)
    emb_pca = pca.transform([emb])
    pred = surrogate.predict(emb_pca)[0]
    label = "AI" if pred == 1 else "Real"
    return {"prediction": label}