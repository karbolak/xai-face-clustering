"""_summary_
    - Hooks activations from layer4 and avgpool (512-2048D)
    - Applies global average pooling to reduce 4D tensors -> 2D
    - Returns a NumPy array of size (N, D) where D ~~ 2048
"""
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1

def get_model(model_name="facenet"):
    if model_name == "facenet":
        model = InceptionResnetV1(pretrained='vggface2').eval()
        layers_to_hook = ["last_linear"]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model, layers_to_hook

def extract_embeddings(images, filenames=None, labels=None, model_name="facenet", cache_path="xai_face_clustering/features/embeddings.npz"):
    if os.path.exists(cache_path):
        print(f"[INFO] Loading cached embeddings from {cache_path}")
        data = np.load(cache_path, allow_pickle=True)
        return data["embeddings"], data["filenames"], data["labels"]

    model, layers_to_hook = get_model(model_name)
    activations = {}

    def get_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    for name, module in model.named_modules():
        if name in layers_to_hook:
            module.register_forward_hook(get_hook(name))

    all_features = []

    with torch.no_grad():
        print("[INFO] Forwarding images through model...")
        for i in tqdm(range(0, len(images), 64)):
            batch = images[i:i+64]
            _ = model(batch)

            batch_feats = []
            for name in layers_to_hook:
                act = activations[name]
                if act.ndim == 4:
                    act = torch.mean(act, dim=[2, 3])
                batch_feats.append(act)
            feats = torch.cat(batch_feats, dim=1)
            all_features.append(feats.cpu())

    embeddings = torch.cat(all_features, dim=0).numpy()

    np.savez(cache_path,
             embeddings=embeddings,
             filenames=np.array(filenames) if filenames is not None else np.arange(len(embeddings)),
             labels=np.array(labels) if labels is not None else np.zeros(len(embeddings)))
    print(f"[INFO] Saved embeddings to {cache_path}")

    return embeddings, filenames, labels