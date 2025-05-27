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
from torchvision import transforms

def get_model(model_name="facenet"):
    if model_name == "facenet":
        model = InceptionResnetV1(pretrained='vggface2').eval()
        layers_to_hook = ["last_linear"]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model, layers_to_hook

# same preprocessing as in streamlit app
IMAGE_SIZE = (224, 224)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def extract_embeddings(images, filenames, labels, model_name, cache_path):
    """
    images: list of np.ndarray in RGB format
    returns: embeddings as (N, 512), plus filenames and labels unchanged
    """
    # 1) load model
    model = InceptionResnetV1(pretrained="vggface2").eval()

    # 2) preprocessing pipeline
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMAGE_SIZE),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    # 3) hook setup
    activations = {}
    def get_hook(_, __, out):
        activations["feat"] = out.detach()
    # register it on the last_linear layer
    for name, module in model.named_modules():
        if name == "last_linear":
            module.register_forward_hook(get_hook)
            break

    # 4) iterate images
    embs = []
    for img in images:
        if isinstance(img, torch.Tensor):
            x = img.unsqueeze(0)
        else:
            x = tf(img).unsqueeze(0)
        with torch.no_grad():
            _ = model(x)
        feat = activations["feat"]
        embs.append(feat.mean(dim=[2,3]).cpu().numpy().squeeze())
    embs = np.vstack(embs)  # (N,512)

    # optional: cache to disk
    np.savez(cache_path, embeddings=embs, filenames=filenames, labels=labels)
    return embs, filenames, labels