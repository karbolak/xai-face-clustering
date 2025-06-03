import streamlit as st
from shap.plots._waterfall import waterfall_legacy
import numpy as np
import joblib
import cv2
import torch
import shap
import matplotlib.pyplot as plt

from facenet_pytorch import InceptionResnetV1
from torchvision import transforms

# ── Config ────────────────────────────
SVM_PATH  = "scripts/xai_face_clustering/models/svm_model.joblib"
PCA_PATH = "scripts/xai_face_clustering/models/pca_model.joblib"
SCALER_PATH = "scripts/xai_face_clustering/models/scaler.joblib"
SHAP_BG_PATH = "scripts/xai_face_clustering/models/shap_background.npz"
background = np.load(SHAP_BG_PATH)["background"]
MODEL_PATH = SVM_PATH  # or LOGREG_PATH
DEFAULT_PCA_COMPONENTS = 100

IMAGE_SIZE    = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
HOOK_LAYER    = "last_linear"

# ── Model Loading ─────────────────────
@st.cache_data(show_spinner=False)
def load_models():
    cnn = InceptionResnetV1(pretrained="vggface2").eval()
    clf = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    pca = joblib.load(PCA_PATH)
    return cnn, scaler, pca, clf

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

# ── Streamlit UI ──────────────────────
st.title("🔍 Real vs. AI-Generated Face Detector")

cnn_model, scaler, pca, clf = load_models()

uploaded = st.file_uploader("Upload a face image", type=["jpg","jpeg","png"], key="face_upload")
if uploaded:
    img_bytes = uploaded.read()
    st.image(img_bytes, caption="Your upload", use_column_width=True)

    with st.spinner("Analyzing…"):
        img_t = preprocess_image(img_bytes)
        emb   = extract_embedding(cnn_model, [HOOK_LAYER], img_t)
        emb_s = scaler.transform([emb])
        emb_p = pca.transform(emb_s)
        pred_label_num = int(clf.predict(emb_p)[0])

        # ---- SHAP: build explainer at runtime ----
        explainer = shap.KernelExplainer(clf.predict_proba, background)
        shap_vals = explainer.shap_values(emb_p)

        st.subheader("SHAP Explanation")
        fig = plt.figure()
        def to_scalar(x):
            return float(x.flat[0]) if isinstance(x, np.ndarray) else float(x)
        if isinstance(shap_vals, list):
            expected_val = to_scalar(explainer.expected_value[pred_label_num])
            svals = shap_vals[pred_label_num][0]
        else:
            expected_val = to_scalar(explainer.expected_value)
            svals = shap_vals[0]
            if svals.ndim == 2:
                # take the column for predicted class
                svals = svals[:, pred_label_num]
        print("expected_val:", expected_val)
        print("svals.shape:", svals.shape)
        print("emb_p[0].shape:", emb_p[0].shape)

        waterfall_legacy(
            expected_val,
            svals,
            emb_p[0],
            feature_names=[f"PC{i+1}" for i in range(emb_p.shape[1])]
        )
        st.pyplot(fig)
        plt.close(fig)


        label = "Real" if pred_label_num == 0 else "AI-Generated"
        st.success(f"**Prediction:** {label}")
