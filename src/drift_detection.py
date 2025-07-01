import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import os
import sys
import decimal

# Ensure the src directory is in the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import aws_database function to get latest production data
from src.aws_database import get_all_production_data

REFERENCE_DATA_PATH = os.getenv("REFERENCE_DATA_PATH", "data/reference/Telco-Customer-Churn.csv")
DRIFT_REPORT_PATH = os.getenv("DRIFT_REPORT_PATH", "drift_report.html")
MIN_NEW_POINTS = 10  # Minimum new data points required for drift detection
LAST_PROCESSED_PATH = "last_processed.txt"

def get_last_processed_date():
    if not os.path.exists(LAST_PROCESSED_PATH):
        return None
    with open(LAST_PROCESSED_PATH, "r") as f:
        return f.read().strip()

def set_last_processed_date(date_str):
    with open(LAST_PROCESSED_PATH, "w") as f:
        f.write(date_str)

def load_reference_data():
    return pd.read_csv(REFERENCE_DATA_PATH)

def convert_decimal_to_float(val):
    if isinstance(val, decimal.Decimal):
        # Try to convert to int if possible, else float
        if val % 1 == 0:
            return int(val)
        else:
            return float(val)
    return val

def load_new_production_data():
    items = get_all_production_data()
    if not items:
        return pd.DataFrame(), None
    items_df = pd.DataFrame(items)
    last_date = get_last_processed_date()
    if last_date:
        # Only use data points with PredictionDate > last processed
        new_items_df = items_df[items_df["PredictionDate"] > last_date]
    else:
        new_items_df = items_df
    if new_items_df.empty or len(new_items_df) < MIN_NEW_POINTS:
        return pd.DataFrame(), None
    # Remove metadata columns for drift detection
    ignore_cols = {"CustomerID", "PredictionDate", "probability"}
    prod_df = new_items_df[[col for col in new_items_df.columns if col not in ignore_cols]]
    # Convert Decimal to float/int
    for col in prod_df.columns:
        prod_df[col] = prod_df[col].map(convert_decimal_to_float)
    # Ensure column types match reference data
    ref_df = pd.read_csv(REFERENCE_DATA_PATH)
    for col in prod_df.columns:
        if col in ref_df.columns:
            try:
                prod_df[col] = prod_df[col].astype(ref_df[col].dtype)
            except Exception:
                pass
    # Get the latest PredictionDate from the new data
    latest_date = new_items_df["PredictionDate"].max()
    return prod_df, latest_date

def run_drift_detection():
    ref = load_reference_data()
    prod, latest_date = load_new_production_data()
    if prod is None or prod.empty:
        print("Not enough new production data for drift detection.")
        return False
    # Align columns
    common_cols = [col for col in ref.columns if col in prod.columns]
    ref = ref[common_cols]
    prod = prod[common_cols]
    # Run drift detection
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=ref, current_data=prod)
    report.save_html(DRIFT_REPORT_PATH)
    drift = report.as_dict()["metrics"][0]["result"]["dataset_drift"]
    print(f"Drift detected: {drift}")
    # Update last processed date
    if latest_date:
        set_last_processed_date(latest_date)
    return drift

if __name__ == "__main__":
    drift = run_drift_detection()
    if drift:
        print("Drift detected! Triggering pipeline...")
        os.system("python run_pipeline.py")