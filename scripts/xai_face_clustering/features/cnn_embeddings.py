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
    images: list of np.ndarray in RGB format or a torch.Tensor of shape (N, 3, 224, 224)
    returns: embeddings as (N, 512), plus filenames and labels unchanged
    """
    # Convert tensor batch to list of numpy arrays if needed
    if isinstance(images, torch.Tensor):
        images = [np.transpose(img.numpy(), (1, 2, 0)) for img in images]

    print(f"[INFO] Extracting embeddings for {len(images)} images...")

    #load model
    model = InceptionResnetV1(pretrained="vggface2").eval()

    #preprocessing pipeline
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMAGE_SIZE),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    #hook setup
    activations = {}
    def get_hook(_, __, out):
        activations["feat"] = out.detach()
    for name, module in model.named_modules():
        if name == "last_linear":
            module.register_forward_hook(get_hook)
            break

    #iterate images with progress bar
    embs = []
    for idx, img in enumerate(tqdm(images, desc="[INFO] Embedding images")):
        if isinstance(img, torch.Tensor):
            x = img.unsqueeze(0)
        else:
            x = tf(img).unsqueeze(0)
        with torch.no_grad():
            _ = model(x)
        feat = activations["feat"]
        embs.append(feat.cpu().numpy().squeeze())
        if (idx + 1) % 500 == 0:
            print(f"[INFO] Processed {idx + 1}/{len(images)} images...")

    embs = np.vstack(embs)  # (N,512)

    print(f"[INFO] Finished embedding extraction. Saving to cache: {cache_path}")
    np.savez(cache_path, embeddings=embs, filenames=filenames, labels=labels)
    return embs, filenames, labels