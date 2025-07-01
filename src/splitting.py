import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger
import os
import mlflow

def split_data(df: pd.DataFrame,
               target_col: str,
               test_size: float = 0.2,
               val_size: float = 0.1,
               random_state: int = 42,
               output_dir: str = None) -> tuple:
    """
    Split df into train/val/test sets in a stratified manner on target_col.
    - test_size: fraction of full data to reserve as test set.
    - val_size: fraction of remaining (after test split) to reserve as validation.
    - random_state: seed.
    - output_dir: if provided, will save CSVs "train.csv", "val.csv", "test.csv" under this directory.
    Returns: X_train, X_val, X_test, y_train, y_val, y_test (as pandas DataFrames/Series).
    """
    logger.info(f"Starting data split with test_size={test_size}, val_size={val_size}, target_col='{target_col}'")
    if target_col not in df.columns:
        logger.error(f"Target column '{target_col}' not found in DataFrame")
        raise KeyError(f"Target column '{target_col}' not found in DataFrame")

    # Separate features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]
    # First split out test set
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    # Then split remaining into train and validation
    # Compute relative validation size w.r.t X_temp
    if not (0 < val_size < 1):
        logger.warning(f"Invalid val_size={val_size}; skipping validation split, setting X_val empty")
        X_train = X_temp
        y_train = y_temp
        X_val = pd.DataFrame(columns=X.columns)
        y_val = pd.Series(dtype=y.dtype)
    else:
        rel_val_size = val_size / (1.0 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=rel_val_size,
            random_state=random_state,
            stratify=y_temp
        )
    logger.info(f"Split result: train={len(X_train)}, val={len(X_val)}, test={len(X_test)}")

    # Optionally save splits
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        # Combine X and y back for saving
        train_df = X_train.copy()
        train_df[target_col] = y_train.values
        val_df = X_val.copy()
        val_df[target_col] = y_val.values
        test_df = X_test.copy()
        test_df[target_col] = y_test.values

        train_path = os.path.join(output_dir, "train.csv")
        val_path = os.path.join(output_dir, "val.csv")
        test_path = os.path.join(output_dir, "test.csv")

        train_df.to_csv(train_path, index=False)
        val_df.to_csv(val_path, index=False)
        test_df.to_csv(test_path, index=False)
        logger.info(f"Saved train/val/test CSVs to '{output_dir}'")

    # MLflow: log splits as artifacts
    with mlflow.start_run(run_name="data_splits", nested=True):
        mlflow.log_artifact(train_path, artifact_path="data_splits")
        mlflow.log_artifact(val_path, artifact_path="data_splits")
        mlflow.log_artifact(test_path, artifact_path="data_splits")

    return X_train, X_val, X_test, y_train, y_val, y_test
