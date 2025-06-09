import os
import cv2
import torch
from torchvision import transforms
from tqdm import tqdm
import numpy as np

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (224, 224)

def load_images(data_dir, as_numpy_list=False):
    """_summary_
        This function will:
        - Traverse the data_dir and read images from both subfolders (e.g., Real_Images, AI-Generated_Images)
        - Resize to 224×224
        - Normalize using ImageNet mean/std
        - Return a list of preprocessed tensors and associated metadata

        Args:
            data_dir (_type_): _description_
            as_numpy_list (bool, optional): If True, return images as a list of numpy arrays (HWC, RGB). Defaults to False.

        Returns:
            images Tensor of shape (N, 3, 224, 224) or List[np.ndarray]
            labels (List[int]): 0 = Real, 1 = AI
            filenames (List[str]): Original filenames
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(IMAGE_SIZE),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ])

    all_images = []
    labels = []
    filenames = []

    label_map = {
        "Real_Images": 0,
        "AI-Generated_Images": 1
        #"real": 0,
        #"fake": 1
    }

    print(f"[INFO] Reading from: {data_dir}")
    for label_folder, label_id in label_map.items():
        folder_path = os.path.join(data_dir, label_folder)
        if not os.path.isdir(folder_path):
            continue
        for fname in tqdm(os.listdir(folder_path), desc=f"Loading {label_folder}"):
            if fname.lower().endswith((".jpg", ".jpeg", ".png")):
                fpath = os.path.join(folder_path, fname)
                img = cv2.imread(fpath)
                if img is None:
                    continue
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img_tensor = transform(img)
                all_images.append(img_tensor)
                labels.append(label_id)
                filenames.append(fname)

    image_batch = torch.stack(all_images)  # shape: (N, 3, 224, 224)
    if as_numpy_list:
        # Convert to list of (224, 224, 3) np.ndarray in RGB
        np_images = [np.transpose(img.numpy(), (1, 2, 0)) for img in image_batch]
        return np_images, labels, filenames
    return image_batch, labels, filenames
