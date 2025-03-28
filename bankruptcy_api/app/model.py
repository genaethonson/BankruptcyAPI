import joblib
import numpy as np
import os

model = joblib.load("model/bankruptcy_model.joblib")
scaler = joblib.load("model/scaler.joblib")
feature_names = joblib.load("model/feature_names.joblib")

def predict(features: list):
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    prob = model.predict_proba(X_scaled)[0][1]
    return int(pred), float(prob)
