import streamlit as st
import numpy as np
import joblib
import json
import os
import cv2
import torch

from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ── Config ────────────────────────────────────────────────────────────────
EMBEDDINGS_CACHE    = "scripts/xai_face_clustering/features/embeddings.npz"
SCALER_PATH         = "scripts/xai_face_clustering/models/scaler.joblib"
PCA_PATH            = "scripts/xai_face_clustering/models/pca_model.joblib"
SURROGATE_PATH      = "scripts/xai_face_clustering/models/surrogate_model.joblib"
CLUSTER_MAP_PATH    = "scripts/xai_face_clustering/models/cluster_label_map.json"

# Must match PCA components used in training
DEFAULT_PCA_COMPONENTS = 100

# CNN embedding settings
IMAGE_SIZE    = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
HOOK_LAYER    = "last_linear"

# ── Model Loading ────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_models():
    cnn = InceptionResnetV1(pretrained="vggface2").eval()
    layers_to_hook = [HOOK_LAYER]

    # Load surrogate & map
    clf = joblib.load(SURROGATE_PATH)
    with open(CLUSTER_MAP_PATH, "r") as f:
        cluster_map = json.load(f)

    # Load scaler & PCA
    scaler, pca = None, None
    try:
        scaler = joblib.load(SCALER_PATH)
        pca    = joblib.load(PCA_PATH)
    except Exception:
        # Ignored: scaler/PCA will be rebuilt on first transform
        pass

    return cnn, layers_to_hook, scaler, pca, clf, cluster_map

# ── Image Preprocessing & Embedding ──────────────────────────────────────
def preprocess_image(image_bytes):
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMAGE_SIZE),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])
    return tf(img)

def extract_embedding(model, hook_layers, img_tensor):
    activations = {}
    hooks = []

    def save_hook(name):
        return lambda _, __, output: activations.setdefault(name, output.detach())

    for name, module in model.named_modules():
        if name in hook_layers:
            hooks.append(module.register_forward_hook(save_hook(name)))

    with torch.no_grad():
        _ = model(img_tensor.unsqueeze(0))

    feats = []
    for name in hook_layers:
        act = activations[name]
        if act.ndim == 4:
            act = act.mean(dim=[2,3])
        feats.append(act)
    emb = torch.cat(feats, dim=1).cpu().numpy()[0]

    for h in hooks:
        h.remove()
    return emb

# ── Streamlit UI ────────────────────────────────────────────────────────
st.title("🔍 Real vs. AI-Generated Face Detector")

cnn_model, hook_layers, scaler, pca, clf, cluster_map = load_models()

uploaded = st.file_uploader("Upload a face image", type=["jpg","jpeg","png"])
if uploaded:
    img_bytes = uploaded.read()
    st.image(img_bytes, caption="Your upload", use_column_width=True)

    with st.spinner("Analyzing…"):
        img_t = preprocess_image(img_bytes)
        emb   = extract_embedding(cnn_model, hook_layers, img_t)

        # Scale & PCA with fallback on mismatch
        try:
            emb_s = scaler.transform([emb])
            emb_p = pca.transform(emb_s)
        except Exception:
            # Rebuild scaler & PCA from embeddings cache
            data = np.load(EMBEDDINGS_CACHE, allow_pickle=True)
            X    = data["embeddings"]
            # Fit new scaler & PCA
            new_scaler = StandardScaler().fit(X)
            Xs         = new_scaler.transform(X)
            new_pca    = PCA(n_components=DEFAULT_PCA_COMPONENTS).fit(Xs)
            # Save over old
            os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
            joblib.dump(new_scaler, SCALER_PATH)
            joblib.dump(new_pca, PCA_PATH)
            # Use new transforms
            emb_s = new_scaler.transform([emb])
            emb_p = new_pca.transform(emb_s)
            # Update in-memory references
            scaler, pca = new_scaler, new_pca

        cluster_id = int(clf.predict(emb_p)[0])
        label_num = cluster_map.get(str(cluster_id), None)
        label     = "Real" if label_num == 0 else "AI-Generated"

    st.success(f"**Prediction:** {label}")
