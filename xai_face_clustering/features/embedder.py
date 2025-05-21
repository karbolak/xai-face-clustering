import os
import pickle
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1

class EmbeddingExtractor:
    def __init__(self,
                 model_name="facenet",
                 cache_path=None,
                 batch_size=64):
        self.cache_path = cache_path
        self.batch_size = batch_size

        # ─── DEVICE SELECTION ────────────────────────────────────────────────
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("⚡ Using MPS backend for embeddings")
        else:
            self.device = torch.device("cpu")
            print("⚡ Using CPU for embeddings")

        # ─── MODEL LOAD + COMPILE + FORMAT ──────────────────────────────────
        if model_name.lower() == "facenet":
            self.model = InceptionResnetV1(pretrained='vggface2') \
                         .eval() \
                         .to(self.device)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        # Try PyTorch 2.0 compile for fusion
        try:
            self.model = torch.compile(self.model)
            print("🔥 Model compiled with torch.compile()")
        except Exception as e:
            print("⚠️  torch.compile() not available:", e)

        # On MPS: switch to NHWC (channels_last) + FP16
        if self.device.type == 'mps':
            self.model = self.model.to(memory_format=torch.channels_last)
            self.model = self.model.half()
            print("⚡ Model set to channels_last & FP16 for MPS")

    def extract(self, dataset):
        # ── 1) Cache load (pickle) ───────────────────────────────────────────
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path, "rb") as f:
                data = pickle.load(f)
            print(f"[EMBED] Loaded embeddings from cache: {self.cache_path}")
            return data["embeddings"], data["names"], data["labels"]

        # ── 2) No cache → batch-wise inference ─────────────────────────────
        num_workers = 4
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=True,
            prefetch_factor=2
        )

        feats, labs, names = [], [], []
        for imgs, labels, fnames in tqdm(loader, desc="[EMBED] Embedding", unit="batch"):
            # Move + cast
            if self.device.type == 'mps':
                imgs = imgs.to(self.device,
                               dtype=torch.float16,
                               memory_format=torch.channels_last)
            else:
                imgs = imgs.to(self.device)

            with torch.no_grad():
                out = self.model(imgs)

            feats.append(out.cpu().numpy())
            labs.extend(labels.tolist())
            names.extend(fnames)

        embeddings = np.vstack(feats)

        # ── 3) Save pickle cache ────────────────────────────────────────────
        if self.cache_path:
            os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump({
                    "embeddings": embeddings,
                    "names":      names,
                    "labels":     labs
                }, f)
            print(f"[EMBED] Saved pickle cache to: {self.cache_path}")

        return embeddings, names, labs
