# xai_face_clustering/features/embedder.py

import os
import sys
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1

class EmbeddingExtractor:
    def __init__(self, model_name="facenet", cache_path=None):
        self.cache_path = cache_path
        if model_name == "facenet":
            # force CPU – avoids any GPU/CUDA binary mismatch
            device = torch.device("cpu")
            self.model = (
                InceptionResnetV1(pretrained='vggface2')
                .eval()
                .to(device)
            )
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

    def extract(self, images, names=None, labels=None):
        if self.cache_path and os.path.exists(self.cache_path):
            data = np.load(self.cache_path, allow_pickle=True)
            print(f"[EMBED] Loaded embeddings from cache: {self.cache_path}")
            return data["embeddings"], data["names"], data["labels"]

        # avoid OpenMP/semaphore issues
        torch.set_num_threads(1)
        torch.multiprocessing.set_sharing_strategy('file_system')

        feats, total = [], len(images)
        print(f"[EMBED] Starting extraction of {total} images in batches of 64")

        with torch.no_grad():
            for start in range(0, total, 64):
                end = min(start + 64, total)
                print(f"[EMBED] → Batch {start}:{end} …", end="", flush=True)
                batch = images[start:end].to('cpu')
                try:
                    out = self.model(batch)
                except Exception as e:
                    print(" ❌ ERROR", file=sys.stderr)
                    print(f"[EMBED]   failed on batch {start}:{end}: {e}", file=sys.stderr)
                    raise
                arr = out.cpu().numpy()
                feats.append(arr)
                print(f" ✓ got {arr.shape}", flush=True)

        print("[EMBED] stacking batches …", end="", flush=True)
        try:
            embeddings = np.vstack(feats)
        except Exception as e:
            print(" ❌ ERROR", file=sys.stderr)
            print(f"[EMBED]   vstack failed: {e}", file=sys.stderr)
            raise
        print(f" ✓ result shape {embeddings.shape}")

        if self.cache_path:
            np.savez(self.cache_path,
                     embeddings=embeddings,
                     names=np.array(names),
                     labels=np.array(labels))
            print(f"[EMBED] saved cache to {self.cache_path}")

        return embeddings, names, labels