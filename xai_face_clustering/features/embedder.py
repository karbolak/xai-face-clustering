import os, numpy as np, torch
from tqdm import tqdm
from facenet_pytorch import InceptionResnetV1

class EmbeddingExtractor:
    def __init__(self, model_name="facenet", cache_path=None):
        self.cache_path = cache_path
        if model_name=="facenet":
            self.model = InceptionResnetV1(pretrained='vggface2').eval()
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")
    def extract(self, images, names=None, labels=None):
        if self.cache_path and os.path.exists(self.cache_path):
            data = np.load(self.cache_path, allow_pickle=True)
            return data["embeddings"], data["names"], data["labels"]
        feats = []
        with torch.no_grad():
            for i in tqdm(range(0,len(images),64)):
                batch = images[i:i+64]
                out = self.model(batch)
                feats.append(out.cpu().numpy())
        embeddings = np.vstack(feats)
        if self.cache_path:
            np.savez(self.cache_path,
                     embeddings=embeddings,
                     names=np.array(names),
                     labels=np.array(labels))
        return embeddings, names, labels
