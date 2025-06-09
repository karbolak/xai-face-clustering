import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans, DBSCAN, AgglomerativeClustering, OPTICS, Birch
from sklearn.mixture import GaussianMixture
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE

try:
    import hdbscan
except ImportError:
    hdbscan = None
try:
    import umap
except ImportError:
    umap = None

import warnings
warnings.filterwarnings('ignore')

#load data
DATA_PATH = "scripts/xai_face_clustering/features/embeddings.npz"
data = np.load(DATA_PATH, allow_pickle=True)
X = data['embeddings']
y_true = np.array(data['labels'])  # 0=Real, 1=AI

#standardize
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("[INFO] Running t-SNE for visualization (may be slow)...")
X_tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000, random_state=42).fit_transform(X_scaled)

if umap:
    print("[INFO] Running UMAP for visualization...")
    X_umap = umap.UMAP(n_components=2, random_state=42).fit_transform(X_scaled)
else:
    X_umap = None

def plot_ground_truth(X_emb, emb_name):
    plt.figure(figsize=(7,6))
    for gt, marker, label in zip([0,1], ['o', 's'], ['Real', 'AI']):
        mask = (y_true == gt)
        plt.scatter(X_emb[mask, 0], X_emb[mask, 1], alpha=0.7, marker=marker, label=label)
    plt.title(f"Ground Truth in {emb_name} space")
    plt.xlabel(f"{emb_name} 1")
    plt.ylabel(f"{emb_name} 2")
    plt.legend()
    plt.tight_layout()
    plt.show()

print("[INFO] Plotting PCA ground truth...")
plot_ground_truth(X_pca, "PCA")
print("[INFO] Plotting t-SNE ground truth...")
plot_ground_truth(X_tsne, "t-SNE")
if X_umap is not None:
    print("[INFO] Plotting UMAP ground truth...")
    plot_ground_truth(X_umap, "UMAP")

def grid_search(estimator, param_grid, X, y_true, mask=None, ari_goal='max', return_model=False):
    from itertools import product
    best_ari = -1
    best_params = None
    best_pred = None
    best_model = None
    keys = list(param_grid.keys())
    for values in product(*param_grid.values()):
        params = dict(zip(keys, values))
        est = estimator(**params)
        try:
            pred = est.fit_predict(X)
        except Exception:
            try:
                est.fit(X)
                pred = est.predict(X)
            except Exception:
                continue
        current_mask = mask(pred) if mask else np.ones_like(pred, dtype=bool)
        if np.sum(current_mask) < 2 or len(set(pred[current_mask])) < 2:
            continue
        ari = adjusted_rand_score(y_true[current_mask], pred[current_mask])
        if ari > best_ari if ari_goal == 'max' else ari < best_ari:
            best_ari = ari
            best_params = params
            best_pred = pred
            if return_model:
                best_model = est
    return best_pred, best_ari, best_params, best_model

results = []

for name, Cls in [("KMeans", KMeans), ("MiniBatchKMeans", MiniBatchKMeans), ("Birch", Birch)]:
    pred, ari, params, _ = grid_search(
        Cls, {'n_clusters': [2, 3, 4, 5, 6]}, X_scaled, y_true
    )
    sil = silhouette_score(X_scaled, pred) if len(set(pred)) > 1 else np.nan
    print(f"{name}: ARI={ari:.2f} (params={params})")
    results.append((name, pred, ari, sil, params))

pred, ari, params, _ = grid_search(
    AgglomerativeClustering, {'n_clusters': [2, 3, 4, 5, 6]}, X_scaled, y_true
)
sil = silhouette_score(X_scaled, pred) if len(set(pred)) > 1 else np.nan
print(f"Agglomerative: ARI={ari:.2f} (params={params})")
results.append(("Agglomerative", pred, ari, sil, params))

def fit_predict_gmm(n_components):
    gmm = GaussianMixture(n_components=n_components, random_state=42)
    return gmm.fit(X_scaled).predict(X_scaled)
best_ari = -1
best_pred = None
best_n = None
for n in [2, 3, 4, 5, 6]:
    pred = fit_predict_gmm(n)
    ari = adjusted_rand_score(y_true, pred)
    if ari > best_ari:
        best_ari = ari
        best_pred = pred
        best_n = n
sil = silhouette_score(X_scaled, best_pred) if len(set(best_pred)) > 1 else np.nan
print(f"GMM: ARI={best_ari:.2f} (n_components={best_n})")
results.append(("GMM", best_pred, best_ari, sil, {'n_components': best_n}))

for name, Cls, param_grid in [
    ("DBSCAN", DBSCAN, {'eps': np.linspace(0.5, 3.0, 8), 'min_samples': [3, 5, 10]}),
    ("OPTICS", OPTICS, {'min_samples': [3, 5, 10], 'xi':[0.05,0.1], 'min_cluster_size':[0.05,0.1,0.2]}),
]:
    def is_clustered(pred): return pred != -1
    pred, ari, params, _ = grid_search(Cls, param_grid, X_scaled, y_true, mask=is_clustered)
    sil = silhouette_score(X_scaled[pred != -1], pred[pred != -1]) if pred is not None and np.any(pred != -1) and len(set(pred[pred != -1])) > 1 else np.nan
    print(f"{name}: ARI={ari:.2f} (params={params})")
    results.append((name, pred, ari, sil, params))

if hdbscan:
    print("[INFO] Running HDBSCAN parameter sweep...")
    best_ari = -1
    best_pred = None
    best_params = None
    for min_cluster_size in [5, 10, 20, 30, 50, 100]:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size)
        pred = clusterer.fit_predict(X_scaled)
        mask = pred != -1
        if np.sum(mask) < 2 or len(set(pred[mask])) < 2: continue
        ari = adjusted_rand_score(y_true[mask], pred[mask])
        if ari > best_ari:
            best_ari = ari
            best_pred = pred
            best_params = {'min_cluster_size': min_cluster_size}
    sil = silhouette_score(X_scaled[best_pred != -1], best_pred[best_pred != -1]) if best_pred is not None and np.any(best_pred != -1) and len(set(best_pred[best_pred != -1])) > 1 else np.nan
    print(f"HDBSCAN: ARI={best_ari:.2f} (params={best_params})")
    results.append(("HDBSCAN", best_pred, best_ari, sil, best_params))

print("\n=== Clustering Leaderboard (by ARI) ===")
results_sorted = sorted(results, key=lambda x: x[2], reverse=True)
for name, _, ari, sil, params in results_sorted:
    print(f"{name}: ARI={ari:.3f}, Silhouette={sil:.3f}, params={params}")

def plot_clusters(X_emb, title, results):
    fig, axes = plt.subplots(2, (len(results)+1)//2, figsize=(18, 10))
    axes = axes.flat
    for ax, (name, pred, ari, sil, params) in zip(axes, results):
        for gt in [0, 1]:
            for cluster in np.unique(pred):
                mask = (y_true == gt) & (pred == cluster)
                ax.scatter(
                    X_emb[mask, 0],
                    X_emb[mask, 1],
                    label=f"True: {'Real' if gt==0 else 'AI'}, Cl: {cluster}",
                    marker="o" if gt==0 else "s",
                    alpha=0.7
                )
        ax.set_title(f"{name}\nARI={ari:.2f}, Sil={sil:.2f}")
        ax.set_xlabel(title+"1")
        ax.set_ylabel(title+"2")
        ax.legend(fontsize="x-small", loc="best", frameon=True, ncol=1)
    plt.tight_layout()
    plt.show()

print("\n[INFO] Plotting clusters in PCA space...")
plot_clusters(X_pca, "PCA", results_sorted[:6])

print("\n[INFO] Plotting clusters in t-SNE space...")
plot_clusters(X_tsne, "tSNE", results_sorted[:6])

if X_umap is not None:
    print("\n[INFO] Plotting clusters in UMAP space...")
    plot_clusters(X_umap, "UMAP", results_sorted[:6])
