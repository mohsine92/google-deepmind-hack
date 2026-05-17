import numpy as np
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

def fit_scaler(X):
    scaler.fit(X)
    return scaler

def transform_features(X):
    return scaler.transform(X)