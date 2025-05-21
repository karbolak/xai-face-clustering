import os
import cv2
from torch.utils.data import Dataset
from torchvision import transforms

class FaceDataset(Dataset):
    """
    Lazily loads images from disk on-the-fly.
    Avoids storing all images in memory at once.
    """
    def __init__(self, data_dir, image_size=(224, 224), mean=None, std=None):
        mean = mean or [0.485, 0.456, 0.406]
        std  = std  or [0.229, 0.224, 0.225]
        # you'll need PIL transformations for Resize etc.
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        self.samples = []
        for label_name, label_id in [("real", 0), ("fake", 1)]:
            folder = os.path.join(data_dir, label_name)
            if not os.path.isdir(folder):
                continue
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg", ".png")):
                    path = os.path.join(folder, fname)
                    self.samples.append((path, label_id, fname))

        if not self.samples:
            raise RuntimeError(f"[LOADER] No images found in {data_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, name = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"[LOADER] Failed to read image: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = self.transform(img)
        return tensor, label, name