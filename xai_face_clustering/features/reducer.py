from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib

class PCAWrapper:
    def __init__(self, n_components, save_path=None):
        self.n_components = n_components
        self.save_path = save_path
    def fit_transform(self, X):
        self.scaler = StandardScaler().fit(X)
        Xs = self.scaler.transform(X)
        self.pca = PCA(n_components=self.n_components).fit(Xs)
        if self.save_path:
            joblib.dump((self.scaler,self.pca), self.save_path)
        return self.pca.transform(Xs)
    def transform(self, X):
        self.scaler, self.pca = joblib.load(self.save_path)
        Xs = self.scaler.transform(X)
        return self.pca.transform(Xs)
