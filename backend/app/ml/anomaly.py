import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class AnomalyModel:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = IsolationForest(
            n_estimators = 200,
            contamination = 0.1,
            random_state = 42
        )
        self.fitted = False

    def fit(self, X):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled)
        self.fitted = True


    def predict(self, features):
                
        X = np.array([[
            features["mean_intensity"],
            features["std_intensity"],
            features["min_intensity"],
            features["max_intensity"],
            features["blur_intensity"]
        ]])

        X_scaled = self.scaler.transform(X)

        score =self.model.decision_function(X_scaled)[0]
        pred = self.model.predict(X_scaled)[0]

        return {
            "anomaly_score": float(score),
            "status": "normal" if pred == 1 else "abnormal"
        }
    
# Initialisation globale du modèle
model =AnomalyModel()