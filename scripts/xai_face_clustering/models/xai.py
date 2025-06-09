import shap
import matplotlib.pyplot as plt
import os

EXPLAIN_DIR = "scripts/xai_face_clustering/features/shap_explanations"
os.makedirs(EXPLAIN_DIR, exist_ok=True)

def run_shap_explanation(model, X, y=None, num_examples=5):
    print("[INFO] Explaining surrogate model using SHAP...")
    print("[INFO] Summarizing background using k-means...")
    background = shap.kmeans(X, 100)

    #kernel explainer for svm, logistic regression and alike
    explainer = shap.KernelExplainer(model.predict_proba, background)
    shap_values = explainer.shap_values(X, nsamples=100)

    for i, class_vals in enumerate(shap_values):
        plt.figure()
        shap.summary_plot(class_vals, X, show=False)
        plt.title(f"SHAP Summary for Class {i}")
        plt.savefig(os.path.join(EXPLAIN_DIR, f"shap_summary_class_{i}.png"))
        plt.close()

    #waterfall plots
    for i in range(min(num_examples, len(X))):
        try:
            plt.figure()
            shap.plots._waterfall.waterfall_legacy(class_vals[i], feature_names=[f"feat_{j}" for j in range(X.shape[1])])
            if y is not None:
                plt.title(f"Sample {i} – True: {y[i]}")
            else:
                plt.title(f"Sample {i}")
            plt.savefig(os.path.join(EXPLAIN_DIR, f"shap_waterfall_{i}.png"))
            plt.close()
        except Exception as e:
            print(f"[WARN] Failed to plot sample {i}: {e}")

    print(f"[INFO] SHAP visualizations saved to {EXPLAIN_DIR}")
