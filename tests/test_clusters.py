import numpy as np
import os

CLUSTER_PATH = "scripts/xai_face_clustering/features/test_clusters.npz"

if not os.path.exists(CLUSTER_PATH):
    raise FileNotFoundError(f"Could not find test cluster file at {CLUSTER_PATH}. "
                            f"Please save y_test_cluster and y_test_orig as .npz after clustering.")

data = np.load(CLUSTER_PATH, allow_pickle=True)
y_test_cluster = data["y_test_cluster"]
y_test_orig = data["y_test_orig"]

unique, counts = np.unique(y_test_cluster, return_counts=True)
print("Test set cluster distribution:")
for cluster_id, count in zip(unique, counts):
    print(f"  Cluster {cluster_id}: {count} samples")

print("\nBreakdown by cluster (Real=0, AI=1):")
for cluster_id in unique:
    idxs = np.where(y_test_cluster == cluster_id)[0]
    gt_labels = np.array(y_test_orig)[idxs]
    real_count = np.sum(gt_labels == 0)
    ai_count = np.sum(gt_labels == 1)
    print(f"  Cluster {cluster_id}: Real={real_count}, AI={ai_count}, Total={len(idxs)}")

print(f"\nTotal test samples: {len(y_test_cluster)}")
print(f"Total 'Real': {np.sum(np.array(y_test_orig)==0)}, Total 'AI': {np.sum(np.array(y_test_orig)==1)}")
