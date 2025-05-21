import os, cv2
import torch
from torchvision import transforms

class ImageLoader:
    def __init__(self, image_size=(224,224), mean=None, std=None):
        mean = mean or [0.485,0.456,0.406]
        std  = std  or [0.229,0.224,0.225]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(image_size),
            transforms.Normalize(mean=mean, std=std)
        ])

    def load(self, data_dir: str):
        images, labels, names = [], [], []
        for label_name, label_id in [("real",0),("fake",1)]:
            folder = os.path.join(data_dir, label_name)
            for fname in os.listdir(folder):
                if fname.lower().endswith((".jpg",".png")):
                    img = cv2.imread(os.path.join(folder,fname))
                    if img is None: continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    images.append(self.transform(img))
                    labels.append(label_id)
                    names.append(fname)
        return torch.stack(images), labels, names
