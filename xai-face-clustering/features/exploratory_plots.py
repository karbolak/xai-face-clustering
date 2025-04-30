import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.feature import local_binary_pattern
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

# Constants
REAL_DIR = 'xai-face-clustering/data/Human_Faces_Dataset/AI-Generated_Images'
FAKE_DIR = 'xai-face-clustering/data/Human_Faces_Dataset/Real_Images'
IMG_SIZE = (128, 128)
LBP_RADIUS = 1
LBP_POINTS = 8 * LBP_RADIUS

OUTPUT_DIR = 'xai-face-clustering/features/exploratory_plots'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Feature lists
features = []
labels = []  # 0 for real, 1 for fake
texture_values = []
edge_densities = []

# Helper functions
def extract_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # LBP for texture
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_POINTS + 3), range=(0, LBP_POINTS + 2), density=True)
    texture_var = np.var(lbp_hist)

    # Canny edge detection
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / (IMG_SIZE[0] * IMG_SIZE[1])

    return lbp_hist, texture_var, edge_density

def process_directory(directory, label):
    for fname in tqdm(os.listdir(directory)):
        fpath = os.path.join(directory, fname)
        img = cv2.imread(fpath)
        if img is not None:
            img = cv2.resize(img, IMG_SIZE)
            lbp_hist, tex_var, edge_dens = extract_features(img)
            features.append(lbp_hist)
            texture_values.append(tex_var)
            edge_densities.append(edge_dens)
            labels.append(label)

# Process images
process_directory(REAL_DIR, 0)
process_directory(FAKE_DIR, 1)

# Normalize and t-SNE
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features)
tsne = TSNE(n_components=2, random_state=42)
tsne_results = tsne.fit_transform(features_scaled)

# Plot 1: t-SNE
plt.figure(figsize=(8, 6))
plt.scatter(tsne_results[np.array(labels) == 0, 0], tsne_results[np.array(labels) == 0, 1], label='Real', alpha=0.6)
plt.scatter(tsne_results[np.array(labels) == 1, 0], tsne_results[np.array(labels) == 1, 1], label='AI-Generated', alpha=0.6)
plt.legend()
plt.title('t-SNE of Texture Features')
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, 'tsne_texture_features.png'))
plt.show()

# Plot 2: Histogram of Texture Variance
plt.figure(figsize=(8, 6))
plt.hist(np.array(texture_values)[np.array(labels) == 0], bins=30, alpha=0.5, label='Real')
plt.hist(np.array(texture_values)[np.array(labels) == 1], bins=30, alpha=0.5, label='AI-Generated')
plt.legend()
plt.title('Histogram of Texture Variance (LBP)')
plt.xlabel('Texture Variance')
plt.ylabel('Count')
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, 'histogram_texture_variance.png'))
plt.show()

# Plot 3: Edge Density Boxplot
plt.figure(figsize=(8, 6))
plt.boxplot([np.array(edge_densities)[np.array(labels) == 0], np.array(edge_densities)[np.array(labels) == 1]],
            labels=['Real', 'AI-Generated'])
plt.title('Edge Density Comparison')
plt.ylabel('Edge Pixel Proportion')
plt.grid(True)
plt.savefig(os.path.join(OUTPUT_DIR, 'boxplot_edge_density.png'))
plt.show()
