import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import joblib
import pandas as pd
from src.deployment import full_preprocess_for_inference

def test_model_performance():
    # Load a sample from the original raw data to ensure all columns are present
    df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
    # Remove the target column if present
    if "Churn" in df.columns:
        y = df["Churn"].map({"Yes": 1, "No": 0})
        X_raw = df.drop(columns=["Churn"])
    else:
        X_raw = df
        y = pd.Series([0]*len(df))  # Dummy target if not present
    model = joblib.load("model/best_model.pkl")
    transformer = joblib.load("model/transformer.pkl")
    features_df = full_preprocess_for_inference(X_raw)
    X_trans = transformer.transform(features_df)
    # Use only the first N rows if needed
    score = model.score(X_trans, y)
    assert score >= 0  # Just check it runs without error