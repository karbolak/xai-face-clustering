# xai_face_clustering/features/loader.py

import os
from PIL import Image
import torch
from torchvision import transforms

class ImageLoader:
    def __init__(self, image_size=(224,224), mean=None, std=None):
        mean = mean or [0.485, 0.456, 0.406]
        std  = std  or [0.229, 0.224, 0.225]
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

    def load(self, data_dir: str):
        print(f"[LOADER] Scanning data directory: {data_dir}")
        images, labels, names = [], [], []

        for label_name, label_id in [("Real_Images",0), ("AI-Generated_Images",1)]:
            folder = os.path.join(data_dir, label_name)
            print(f"[LOADER] Processing folder: {folder} (label {label_id})")
            try:
                files = [f for f in os.listdir(folder) if f.lower().endswith((".jpg",".png"))]
            except Exception as e:
                print(f"[LOADER]   ERROR listing {folder}: {e}")
                continue
            print(f"[LOADER]   Found {len(files)} image files")

            for fname in files:
                path = os.path.join(folder, fname)
                print(f"[LOADER]   Loading {path}", end="", flush=True)
                try:
                    img = Image.open(path).convert("RGB")
                    tensor = self.transform(img)
                    images.append(tensor)
                    labels.append(label_id)
                    names.append(fname)
                    print(f" → success (tensor shape {tensor.shape})")
                except Exception as e:
                    print(f" → ERROR: {e}")

        if not images:
            raise RuntimeError(f"[LOADER] No images loaded from {data_dir}")

        print(f"[LOADER] Stacking {len(images)} tensors into batch")
        try:
            batch = torch.stack(images)
        except Exception as e:
            print(f"[LOADER] ERROR stacking tensors: {e}")
            raise
        print(f"[LOADER] batch shape: {batch.shape}")

        return batch, labels, names