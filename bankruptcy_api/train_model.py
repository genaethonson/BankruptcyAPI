# train_model.py
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
import joblib
import os

os.makedirs("model", exist_ok=True)

def train_model():
    df = pd.read_csv("data.csv")
    y = df["Bankrupt?"]
    X = df.drop(columns=["Bankrupt?"])
    
#top 15 feature based on correlation
    correlation = X.corrwith(y).abs().sort_values(ascending=False)
    top_features = correlation.head(15).index.tolist()
    X = X[top_features]
#using stratification to preserve the class distribution when splitting the data. ex. 94% not bankrupt, 6% bankrupt in the pool 
#otheriwse can split in a way where the test set contains mostly one class. stratificaion make sure the train and test set maintains the same 94/6 class ratio


    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    weights = compute_class_weight(class_weight='balanced', classes=np.array([0, 1]), y=y_train)
    model = LogisticRegression(class_weight={0: weights[0], 1: weights[1]}, max_iter=1000)
    model.fit(X_train_scaled, y_train)

    joblib.dump(model, "model/bankruptcy_model.joblib")
    joblib.dump(scaler, "model/scaler.joblib")
    joblib.dump(top_features, "model/feature_names.joblib")

    print("✅ Model trained and saved.")

if __name__ == "__main__":
    train_model()
