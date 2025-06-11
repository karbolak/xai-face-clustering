import streamlit as st
import numpy as np
import joblib
import json
import os
import cv2
import torch
import shap
import matplotlib.pyplot as plt

from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

#hardcoded values
EMBEDDINGS_CACHE    = "scripts/xai_face_clustering/features/embeddings.npz"
SCALER_PATH         = "scripts/xai_face_clustering/models/scaler.joblib"
PCA_PATH            = "scripts/xai_face_clustering/models/pca_model.joblib"
SURROGATE_PATH      = "scripts/xai_face_clustering/models/surrogate_model.joblib"
CLUSTER_MAP_PATH    = "scripts/xai_face_clustering/models/cluster_label_map.json"
from PIL import Image

st.set_page_config(
    page_title="Face Clustering with XAI",
    page_icon="🧠",
    layout="wide"
)

st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;700&display=swap" rel="stylesheet">
    <style>
        html, body, [class*="css"] {
            font-family: 'Inter', sans-serif;
            color: #000000 !important;
        }
        .stApp {
            background-color: #ffffff;
            padding: 2rem;
        }
        h1 {
            text-align: center;
            font-size: 3em;
            margin-bottom: 0.2em;
        }
        .resultBox {
            text-align: center;
            font-size: 26px;
            padding: 20px;
            background-color: #f8f9fa;
            border-radius: 15px;
            box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
            margin-top: 20px;
        }
        .footer {
            text-align: center;
            color: #000000 !important;
            margin-top: 3rem;
            font-size: 0.85rem;
        }
    </style>
""", unsafe_allow_html=True)

DEFAULT_PCA_COMPONENTS = 100

IMAGE_SIZE    = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
HOOK_LAYER    = "last_linear"

@st.cache_data(show_spinner=False)
def load_models():
    cnn = InceptionResnetV1(pretrained="vggface2").eval()
    layers_to_hook = [HOOK_LAYER]

    clf = joblib.load(SURROGATE_PATH)
    with open(CLUSTER_MAP_PATH, "r") as f:
        cluster_map = json.load(f)

    scaler, pca = None, None
    try:
        scaler = joblib.load(SCALER_PATH)
        pca    = joblib.load(PCA_PATH)
    except Exception:
        pass

    return cnn, layers_to_hook, scaler, pca, clf, cluster_map

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

st.markdown("<h1>🤖 Real vs. AI-Generated Face Detector</h1>", unsafe_allow_html=True)

cnn_model, hook_layers, scaler, pca, clf, cluster_map = load_models()

uploaded = st.file_uploader("Upload a face image ☞", type=["jpg", "jpeg", "png"])
if uploaded:
    img_bytes = uploaded.read()
    st.image(img_bytes, caption="🖼️ Your Uploaded Image", use_container_width=True)

    with st.spinner("☕︎ Analyzing…"):
        img_t = preprocess_image(img_bytes)
        emb   = extract_embedding(cnn_model, hook_layers, img_t)

        try:
            emb_s = scaler.transform([emb])
            emb_p = pca.transform(emb_s)
        except Exception:
            data = np.load(EMBEDDINGS_CACHE, allow_pickle=True)
            X    = data["embeddings"]
            new_scaler = StandardScaler().fit(X)
            Xs         = new_scaler.transform(X)
            new_pca    = PCA(n_components=DEFAULT_PCA_COMPONENTS).fit(Xs)

            os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
            joblib.dump(new_scaler, SCALER_PATH)
            joblib.dump(new_pca, PCA_PATH)
            
            emb_s = new_scaler.transform([emb])
            emb_p = new_pca.transform(emb_s)
            
            scaler, pca = new_scaler, new_pca

        cluster_id = int(clf.predict(emb_p)[0])
        label_num = cluster_map.get(str(cluster_id), None)
        label = "Real" if label_num == 0 else "AI-Generated"

        st.markdown(f"""
        <div class='resultBox'>
            ☻ <strong>Prediction: {label}</strong><br>
            This face appears to be <strong>{label}</strong>.
        </div>
        """, unsafe_allow_html=True)

        
        show_shap = st.checkbox("⌚︎ Show SHAP Explanation (may be slow)", value=False)
        if show_shap:
            if not os.path.exists(EMBEDDINGS_CACHE):
                st.error("No embeddings cache found for SHAP background.")
            else:
                data = np.load(EMBEDDINGS_CACHE, allow_pickle=True)
                X_bg = data["embeddings"]
                X_bg_s = scaler.transform(X_bg)
                X_bg_p = pca.transform(X_bg_s)

                K = 20
                bg_summary = shap.kmeans(X_bg_p, K)

                with st.spinner("Computing SHAP..."):
                    try:
                        explainer = shap.KernelExplainer(clf.predict_proba, bg_summary)
                        shap_vals = explainer.shap_values(emb_p)

                        idx = label_num if label_num in [0, 1] else 0
                        if isinstance(shap_vals, list):
                            sv = np.array(shap_vals[idx][0])
                            ev = explainer.expected_value[idx]
                        else:
                            sv = np.array(shap_vals[0])
                            ev = explainer.expected_value

                        if sv.ndim > 1:
                            sv = sv.squeeze()
                        if sv.ndim > 1:
                            sv = sv.ravel()[:emb_p.shape[1]]

                        if isinstance(ev, (np.ndarray, list)) and len(np.array(ev).shape) > 0:
                            ev = float(np.array(ev).flatten()[0])

                        shap_exp = shap.Explanation(
                            values=sv,
                            base_values=ev,
                            data=emb_p[0],
                            feature_names=[f'PC{i+1}' for i in range(emb_p.shape[1])]
                        )

                        fig = plt.figure()
                        shap.plots.waterfall(shap_exp, show=False)
                        st.pyplot(fig)
                        plt.close(fig)
                    except Exception as e:
                        st.error(f"SHAP computation failed: {e}")


st.markdown("<hr>", unsafe_allow_html=True)
st.markdown("<div class='footer'>© 2025 XAI Face Detector • Built by RUG Students using Streamlit</div>", unsafe_allow_html=True)