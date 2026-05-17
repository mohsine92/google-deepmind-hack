import numpy as np
from sklearn.ensemble import IsolationForest


model = IsolationForest(
    n_estimators= 100,
    contamination =0.1,
    random_state = 42
)

# fake trainng data (simulation microscopy baseline)
def train_dummy_model():
    X = np.random.normal(0,1, (200, 4))
    model.fit(X)

train_dummy_model()


def predict_anomaly( features: dict):
    X = np.array([[
        features["mean_intensity"],
        features["std_intensity"],
        features["min_intensity"],
        features["max_intensity"]
    ]])

    score = model.decision_function(X)[0]
    pred = model.predict(X)[0]

    return {
        "anomaly_score": float(score),
        "status": 'normal' if pred == 1 else 'abnormal'
    }