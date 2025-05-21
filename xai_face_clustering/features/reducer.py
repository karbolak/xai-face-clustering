from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pickle
import os

class PCAWrapper:
    def __init__(self, n_components, save_path=None):
        self.n_components = n_components
        self.save_path     = save_path

    def fit_transform(self, X):
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.pca    = PCA(n_components=self.n_components).fit(Xs)

        if self.save_path:
            d = os.path.dirname(self.save_path)
            if d:
                os.makedirs(d, exist_ok=True)
            with open(self.save_path, 'wb') as f:
                pickle.dump((self.scaler, self.pca), f)

        return self.pca.transform(Xs)

    def transform(self, X):
        with open(self.save_path, 'rb') as f:
            self.scaler, self.pca = pickle.load(f)
        Xs = self.scaler.transform(X)
        return self.pca.transform(Xs)
