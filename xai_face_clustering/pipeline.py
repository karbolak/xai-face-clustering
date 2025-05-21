# xai_face_clustering/pipeline.py

import os
import shap
import numpy as np
from sklearn.model_selection import train_test_split

from .data.loader       import ImageLoader
from .features.embedder import EmbeddingExtractor
from .features.reducer  import PCAWrapper
from .models.clustering import KMeansCluster, DBSCANCluster
from .models.surrogate  import SurrogateModel
from .models.explain    import ShapExplainer

class PipelineRunner:
    def __init__(self, cfg):
        print("[INIT] PipelineRunner configuration:")
        for k, v in vars(cfg).items():
            print(f"       • {k} = {v}")

        self.loader = ImageLoader(cfg.IMG_SIZE, cfg.MEAN, cfg.STD)
        print(f"[INIT] Created ImageLoader(size={cfg.IMG_SIZE})")

        self.embedder = EmbeddingExtractor(model_name=cfg.MODEL_NAME,
                                           cache_path=cfg.CACHE)
        print(f"[INIT] Created EmbeddingExtractor(model={cfg.MODEL_NAME})")

        self.pca = PCAWrapper(cfg.PCA_COMPONENTS, cfg.PCA_SAVE)
        print(f"[INIT] Created PCAWrapper(n_components={cfg.PCA_COMPONENTS})")

        if cfg.CLUSTER_METHOD.lower() == "dbscan":
            self.cluster = DBSCANCluster(eps=cfg.EPS, min_samples=cfg.MIN_SAMPLES)
            print(f"[INIT] Using DBSCANCluster(eps={cfg.EPS}, min_samples={cfg.MIN_SAMPLES})")
        else:
            self.cluster = KMeansCluster(k=cfg.K)
            print(f"[INIT] Using KMeansCluster(k={cfg.K})")

        self.surrogate = SurrogateModel(method=cfg.SURROGATE)
        print(f"[INIT] SurrogateModel(method={cfg.SURROGATE}) ready")

        os.makedirs(cfg.XAI_OUT, exist_ok=True)
        self.xai_out = cfg.XAI_OUT
        print(f"[INIT] SHAP outputs directory: {self.xai_out}")

        self.cfg = cfg

    def run(self):
        print("\n[RUN] === Starting pipeline ===")

        # 1) Load
        print(f"[RUN] 1/6 Loading images from {self.cfg.DATA_DIR} …")
        imgs, true_labels, names = self.loader.load(self.cfg.DATA_DIR)
        print(f"[RUN]    Loaded {len(imgs)} images, {len(set(true_labels))} classes")

        # 2) Embedding
        print("[RUN] 2/6 Extracting embeddings …")
        embeddings, names, true_labels = self.embedder.extract(imgs, names, true_labels)
        print(f"[RUN]    Embeddings shape: {embeddings.shape}")

        # 3) PCA
        print("[RUN] 3/6 Applying PCA …")
        Xp = self.pca.fit_transform(embeddings)
        print(f"[RUN]    PCA output shape: {Xp.shape}")
        if hasattr(self.pca, "explained_variance_ratio_"):
            evr = self.pca.explained_variance_ratio_.sum()
            print(f"[RUN]    Explained variance sum: {evr:.3f}")

        # 4) Clustering
        print("[RUN] 4/6 Clustering …")
        pred_labels = self.cluster.fit_predict(Xp)
        ncl = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
        print(f"[RUN]    Found {ncl} clusters")
        cluster_metrics = self.cluster.evaluate(Xp, true_labels)
        print(f"[RUN]    Cluster metrics: {cluster_metrics}")

        # 5) Train/Test Split
        print("[RUN] 5/6 Splitting for surrogate …")
        X_train, X_test, y_train, y_test = train_test_split(
            Xp, pred_labels, test_size=self.cfg.TEST_SIZE, random_state=42
        )
        print(f"[RUN]    Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

        # 6) Surrogate
        print("[RUN] 6/6 Training surrogate …")
        self.surrogate.train(X_train, y_train)
        rep = self.surrogate.evaluate(X_test, y_test)
        print("[RUN]    Surrogate report summary:")
        for cls, m in rep.items():
            if isinstance(m, dict):
                print(f"       • {cls}: precision={m['precision']:.2f}, recall={m['recall']:.2f}, f1={m['f1-score']:.2f}")

        # 7) SHAP
        print("[RUN] Generating SHAP explanations …")
        bg = shap.kmeans(X_train, min(100, X_train.shape[0]))
        self.explainer = ShapExplainer(self.surrogate.model, bg, self.xai_out)

        print("[RUN] • Global summary …")
        summary = self.explainer.global_summary(X_test)
        print(f"[RUN]   Saved summary: {summary}")

        print(f"[RUN] • Local waterfalls (n={self.cfg.N_WATERFALLS}) …")
        wf = self.explainer.local_waterfalls(X_test, n=self.cfg.N_WATERFALLS)
        for p in wf:
            print(f"[RUN]   Saved waterfall: {p}")

        print("[RUN] === Pipeline complete ===\n")
        return {
            "cluster_metrics":  cluster_metrics,
            "surrogate_report": rep,
            "summary_plot":     summary,
            "waterfall_plots":  wf
        }