import os
import shap
import matplotlib.pyplot as plt

class ShapExplainer:
    def __init__(self, model, background, out_dir):
        """
        :param model: trained surrogate (must have .predict)
        :param background: numpy array for SHAP background
        :param out_dir: where to save PNGs
        """
        self.model = model
        self.background = background
        self.out_dir = out_dir
        os.makedirs(self.out_dir, exist_ok=True)
        self.explainer = shap.KernelExplainer(self.model.predict, self.background)

    def global_summary(self, X):
        shap_vals = self.explainer.shap_values(X)
        plt.figure()
        shap.summary_plot(shap_vals, X, show=False)
        path = os.path.join(self.out_dir, "shap_summary_plot.png")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path

    def local_waterfalls(self, X, n=5):
        shap_vals = self.explainer.shap_values(X[:n])
        # for binary classifiers shap_vals is a list [neg, pos]
        vals = shap_vals[1] if isinstance(shap_vals, list) else shap_vals
        feature_names = [f"PC{i+1}" for i in range(X.shape[1])]
        paths = []
        for i in range(min(n, vals.shape[0])):
            plt.figure()
            shap.plots._waterfall.waterfall_legacy(
                self.explainer.expected_value,
                vals[i],
                feature_names=feature_names
            )
            p = os.path.join(self.out_dir, f"waterfall_{i}.png")
            plt.tight_layout()
            plt.savefig(p)
            plt.close()
            paths.append(p)
        return paths
