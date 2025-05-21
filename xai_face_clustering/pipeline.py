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
        print("[INIT] Building pipeline with config:")
        for k, v in vars(cfg).items():
            print(f"       • {k} = {v}")
        # loader & embeddings
        self.loader    = ImageLoader(cfg.IMG_SIZE, cfg.MEAN, cfg.STD)
        print(f"[INIT] Created ImageLoader(size={cfg.IMG_SIZE})")

        self.embedder  = EmbeddingExtractor(model_name=cfg.MODEL_NAME,
                                            cache_path=cfg.CACHE)
        print(f"[INIT] Created EmbeddingExtractor(model={cfg.MODEL_NAME})")

        # dimensionality reduction
        self.pca       = PCAWrapper(cfg.PCA_COMPONENTS, cfg.PCA_SAVE)
        print(f"[INIT] Created PCAWrapper(components={cfg.PCA_COMPONENTS})")

        # clustering (choose KMeans or DBSCAN)
        if cfg.CLUSTER_METHOD.lower() == "dbscan":
            self.cluster = DBSCANCluster(cfg.EPS, cfg.MIN_SAMPLES)
            print(f"[INIT] Using DBSCANCluster(eps={cfg.EPS}, min_samples={cfg.MIN_SAMPLES})")
        else:
            self.cluster = KMeansCluster(cfg.K)
            print(f"[INIT] Using KMeansCluster(k={cfg.K})")

        # surrogate & explainer
        self.surrogate = SurrogateModel(method=cfg.SURROGATE)
        print(f"[INIT] SurrogateModel(method={cfg.SURROGATE}) ready")

        # XAI outputs
        os.makedirs(cfg.XAI_OUT, exist_ok=True)
        self.xai_out = cfg.XAI_OUT
        print(f"[INIT] XAI outputs will be saved to: {self.xai_out}")

        self.cfg = cfg

    def run(self):
        print("\n[RUN] === Starting pipeline ===")

        # 1) load & embed
        print(f"[RUN] 1/6 Loading images from {self.cfg.DATA_DIR} …")
        imgs, true_labels, names = self.loader.load(self.cfg.DATA_DIR)
        print(f"[RUN]    Loaded {len(imgs)} images, {len(set(true_labels))} classes")

        print("[RUN] 2/6 Extracting embeddings …")
        embeddings, names, true_labels = self.embedder.extract(imgs, names, true_labels)
        print(f"[RUN]    Embeddings shape: {embeddings.shape}")

        # 2) PCA
        print("[RUN] 3/6 Applying PCA …")
        Xp = self.pca.fit_transform(embeddings)
        print(f"[RUN]    PCA output shape: {Xp.shape}")
        if hasattr(self.pca, 'explained_variance_ratio_'):
            evr = self.pca.explained_variance_ratio_.sum()
            print(f"[RUN]    Explained variance (sum): {evr:.3f}")

        # 3) cluster
        print("[RUN] 4/6 Clustering …")
        pred_labels = self.cluster.fit_predict(Xp)
        print(f"[RUN]    Assigned to {len(set(pred_labels))} clusters "
              f"(including noise)" if -1 in pred_labels else "")
        cluster_metrics = self.cluster.evaluate(Xp, true_labels)
        print(f"[RUN]    Cluster metrics: {cluster_metrics}")

        # 4) train/test split on predicted clusters
        print("[RUN] 5/6 Splitting data for surrogate …")
        X_train, X_test, y_train, y_test = train_test_split(
            Xp, pred_labels, test_size=self.cfg.TEST_SIZE, random_state=42
        )
        print(f"[RUN]    Train size: {X_train.shape[0]}  Test size: {X_test.shape[0]}")

        # 5) surrogate
        print("[RUN] 6/6 Training surrogate model …")
        self.surrogate.train(X_train, y_train)
        surrogate_report = self.surrogate.evaluate(X_test, y_test)
        print("[RUN]    Surrogate classification report:")
        for cls, metrics in surrogate_report.items():
            if isinstance(metrics, dict):
