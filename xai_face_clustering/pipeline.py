import os
import shap
import numpy as np
from sklearn.model_selection import train_test_split

from .data.loader       import FaceDataset
from .features.embedder import EmbeddingExtractor
from .features.reducer  import PCAWrapper
from .models.clustering import KMeansCluster, DBSCANCluster
from .models.surrogate  import SurrogateModel
from .models.explain    import ShapExplainer

class PipelineRunner:
    def __init__(self, cfg):
        print("[INIT] PipelineRunner config:")
        for k, v in vars(cfg).items():
            print(f"   • {k} = {v}")
        self.cfg = cfg

        # Dataset & embedding
        self.dataset  = FaceDataset(cfg.DATA_DIR, cfg.IMG_SIZE, cfg.MEAN, cfg.STD)
        self.embedder = EmbeddingExtractor(
            model_name=cfg.MODEL_NAME,
            cache_path=cfg.CACHE,
            batch_size=cfg.BATCH_SIZE
        )

        # PCA, clustering, surrogate, SHAP dir…
        self.pca    = PCAWrapper(cfg.PCA_COMPONENTS, cfg.PCA_SAVE)
        self.cluster = (
            DBSCANCluster(eps=cfg.EPS, min_samples=cfg.MIN_SAMPLES)
            if cfg.CLUSTER_METHOD.lower() == "dbscan"
            else KMeansCluster(k=cfg.K)
        )
        self.surrogate = SurrogateModel(cfg.SURROGATE)
        os.makedirs(cfg.XAI_OUT, exist_ok=True)
        self.xai_out = cfg.XAI_OUT

    def run(self):
        print("\n[RUN] === Starting pipeline ===")

        # 1) Embed (or load cache)
        embeddings, names, true_labels = self.embedder.extract(self.dataset)
        print(f"[RUN] 1/7 Embeddings shape: {embeddings.shape}")

        # 2) PCA
        print("[RUN] 2/7 Applying PCA")
        Xp = self.pca.fit_transform(embeddings)
        print(f"[RUN]    PCA → {Xp.shape}")

        # 3) Clustering
        print("[RUN] 3/7 Clustering")
        pred_labels = self.cluster.fit_predict(Xp)
        ncl = len(set(pred_labels)) - (1 if -1 in pred_labels else 0)
        print(f"[RUN]    Found {ncl} clusters")
        cm = self.cluster.evaluate(Xp, true_labels)

        # 4) Surrogate split/train
        print("[RUN] 4/7 Training surrogate")
        Xtr, Xte, ytr, yte = train_test_split(
            Xp, pred_labels, test_size=self.cfg.TEST_SIZE, random_state=42
        )
        self.surrogate.train(Xtr, ytr)
        rep = self.surrogate.evaluate(Xte, yte)

        # 5) SHAP
        print("[RUN] 5/7 Generating SHAP plots")
        bg = shap.kmeans(Xtr, min(100, len(Xtr)))
        expl = ShapExplainer(self.surrogate.model, bg, self.xai_out)
        summary = expl.global_summary(Xte)
        wfets   = expl.local_waterfalls(Xte, n=self.cfg.N_WATERFALLS)

        print("[RUN] === Pipeline complete ===")
        return {
            "cluster_metrics":  cm,
            "surrogate_report": rep,
            "summary_plot":     summary,
            "waterfall_plots":  wfets
        }