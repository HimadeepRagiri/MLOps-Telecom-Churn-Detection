import os
import pandas as pd
from loguru import logger

def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Load raw CSV data into a DataFrame.
    - file_path: path to raw CSV, e.g., 'data/raw/Telco-Customer-Churn.csv'.
    Return: DataFrame.
    """
    logger.info(f"Loading raw data from {file_path}")
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"Raw data file not found at {file_path}")
    df = pd.read_csv(file_path)
    logger.info(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

def save_dataframe(df: pd.DataFrame, output_path: str):
    """
    Save DataFrame to CSV at output_path, creating directories if needed.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved DataFrame to {output_path}")
