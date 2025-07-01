import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd

def test_raw_data_schema():
    df = pd.read_csv("data/raw/Telco-Customer-Churn.csv")
    expected_cols = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
        "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
        "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
        "MonthlyCharges", "TotalCharges", "Churn"
    ]
    assert all(col in df.columns for col in expected_cols)

def test_no_missing_in_processed():
    df = pd.read_csv("data/processed/train.csv")
    assert df.isna().sum().sum() == 0