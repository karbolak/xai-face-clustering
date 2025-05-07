import shap
import numpy as np
import matplotlib.pyplot as plt
import os

EXPLAIN_DIR = "xai-face-clustering/features/shap_explanations"
os.makedirs(EXPLAIN_DIR, exist_ok=True)

def run_shap_explanation(model, X, cluster_labels, num_examples=5):
    """
        Generates SHAP explanations for a trained surrogate model.
        
        Global view: summary_plot shows most important features across all data
        Local view: waterfall plots explain individual predictions
        Portable: Saved as .png so they can be reviewed without launching an interactive viewer

        Args:
            model: Trained scikit-learn classifier (LogReg / Tree)
            X (np.ndarray): Feature vectors used for training
            cluster_labels (np.ndarray): Corresponding cluster labels
            num_examples (int): Number of individual examples to explain
    """

    print("[INFO] Explaining surrogate model using SHAP...")
    explainer = shap.Explainer(model, X)
    shap_values = explainer(X)

    # Summary plot (global)
    plt.figure()
    shap.summary_plot(shap_values, X, show=False)
    plt.title("SHAP Summary Plot")
    plt.savefig(os.path.join(EXPLAIN_DIR, "shap_summary_plot.png"))
    plt.close()

    # Local explanations for individual examples
    for i in range(min(num_examples, len(X))):
        plt.figure()
        shap.plots.waterfall(shap_values[i], show=False)
        plt.title(f"Image {i} → Cluster {cluster_labels[i]}")
        plt.savefig(os.path.join(EXPLAIN_DIR, f"shap_explanation_{i}.png"))
        plt.close()

    print(f"[INFO] SHAP visualizations saved to {EXPLAIN_DIR}")
