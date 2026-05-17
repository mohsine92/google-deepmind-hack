import numpy as np
from sklearn.ensemble import IsolationForest
from backend.app.datasets.bbbc_loader import load_bbbc_features

class BBBCAnomalyModel:

    def __init__(self):
        self.model = IsolationForest(
            n_estimators=200,
            contamination=0.1,
            random_state=42
        )
        self.fitted = False

    def train(self):
        dataset = load_bbbc_features(limit=50)

        X = np.array([
            [
                d["mean_intensity"],
                d["std_intensity"],
                d["min_intensity"],
                d["max_intensity"],
                d["blur_intensity"]
            ]
            for d in dataset
        ])

        self.model.fit(X)
        self.fitted = True

    def predict(self, features):

        X = np.array([[
            features["mean_intensity"],
            features["std_intensity"],
            features["min_intensity"],
            features["max_intensity"],
            features["blur_intensity"]
        ]])

        score = self.model.decision_function(X)[0]
        pred = self.model.predict(X)[0]

        return {
            "anomaly_score": float(score),
            "status": "normal" if pred == 1 else "abnormal"
        }


# instance globale
model = BBBCAnomalyModel()
model.train()