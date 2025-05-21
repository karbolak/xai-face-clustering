from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report

class SurrogateModel:
    def __init__(self, method="logreg"):
        if method == "logreg":
            self.model = LogisticRegression(max_iter=1000)
        elif method == "tree":
            self.model = DecisionTreeClassifier(max_depth=3)
        elif method == "svm":
            self.model = SVC(kernel="rbf", probability=True)
        else:
            raise ValueError(f"Unknown surrogate method: {method}")

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        # return as dict so tests can inspect metrics
        return classification_report(y_test, y_pred, output_dict=True)
