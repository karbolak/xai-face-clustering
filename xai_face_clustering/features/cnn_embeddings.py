"""_summary_
    - Hooks activations from layer4 and avgpool (512-2048D)
    - Applies global average pooling to reduce 4D tensors -> 2D
    - Returns a NumPy array of size (N, D) where D ~~ 2048
"""

import torch
import torchvision.models as models
import torch.nn as nn
from tqdm import tqdm

def get_model(model_name="resnet50"):
    if model_name == "resnet50":
        model = models.resnet50(pretrained=True)
        layers_to_hook = ["layer4", "avgpool"]
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=True)
        layers_to_hook = ["features"]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    model.eval()
    return model, layers_to_hook


def extract_embeddings(images, model_name="resnet50"):
    model, layers_to_hook = get_model(model_name)
    activations = {}

    def get_hook(name):
        def hook(module, input, output):
            activations[name] = output.detach()
        return hook

    # Register forward hooks
    for name, module in model.named_modules():
        if name in layers_to_hook:
            module.register_forward_hook(get_hook(name))

    with torch.no_grad():
        print("[INFO] Forwarding images through model...")
        for i in tqdm(range(0, len(images), 64)):  # batch size 64
            batch = images[i:i+64]
            _ = model(batch)

    # Process collected activations
    flattened = []
    for name in layers_to_hook:
        act = activations[name]
        if act.ndim == 4:
            act = torch.mean(act, dim=[2, 3])  # Global average pool (spatial)
        flattened.append(act)

    features = torch.cat(flattened, dim=1)  # Shape: (N, D)
    return features.numpy()
