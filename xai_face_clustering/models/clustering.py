from abc import ABC, abstractmethod
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
from xai_face_clustering.features.dbscan import apply_dbscan

class ClusteringStrategy(ABC):
    @abstractmethod
    def fit_predict(self, X):
        """Fit the clustering model and return labels"""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, X, true_labels):
        """Return a dict with silhouette and ARS"""
        raise NotImplementedError

class KMeansCluster(ClusteringStrategy):
    def __init__(self, k):
        self.k = k

    def fit_predict(self, X):
        self.model = KMeans(n_clusters=self.k, random_state=42).fit(X)
        self.labels = self.model.labels_
        return self.labels

    def evaluate(self, X, true_labels):
        sil = silhouette_score(X, self.labels) if len(set(self.labels)) > 1 else -1
        ars = adjusted_rand_score(true_labels, self.labels)
        return {"silhouette_score": sil, "adjusted_rand_score": ars}

class DBSCANCluster(ClusteringStrategy):
    def __init__(self, eps, min_samples):
        self.eps = eps
        self.min_samples = min_samples

    def fit_predict(self, X):
        print("DBSCAN CLUSTER fit_predict")
        self.model = apply_dbscan(X, eps=self.eps, min_samples=self.min_samples)
        self.labels = self.model.labels_
        return self.labels

    def evaluate(self, X, true_labels):
        n_clusters = len(set(self.labels)) - (1 if -1 in self.labels else 0)
        sil = silhouette_score(X, self.labels) if n_clusters > 1 else -1
        ars = adjusted_rand_score(true_labels, self.labels)
        return {"silhouette_score": sil, "adjusted_rand_score": ars}
