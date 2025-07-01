from zenml import pipeline, step
import pandas as pd
import os
import joblib
from loguru import logger
import mlflow

# Import project modules
from src.data_ingestion import load_raw_data
from src.preprocessing import (
    convert_object_to_numeric, encode_binary_columns, handle_missing_values,
    handle_duplicates, standardize_formats, handle_outliers,
    drop_irrelevant_features, convert_dtypes, scale_and_encode
)
from src.feature_engineering import (
    create_ratio_features, create_interaction_features,
    drop_high_cardinality_categorical, bin_numerical_feature
)
from src.splitting import split_data
from src.training import train_and_log_all_models
from src.evaluation import evaluate_all_models_on_test

# -----------------------
# Step 1: Ingestion
@step
def ingest_step(raw_data_path: str) -> pd.DataFrame:
    """
    ZenML step: load raw CSV into DataFrame.
    """
    df = load_raw_data(raw_data_path)
    return df

# -----------------------
# Step 2: Preprocessing & Feature Engineering
@step
def preprocessing_step(
    df: pd.DataFrame,
    # Config parameters (to be supplied from run_pipeline.py)
    object_numeric_cols: list,
    binary_mapping: dict,
    missing_numeric_cols: list,
    missing_categorical_cols: list,
    drop_duplicate_subset: list,
    datetime_cols: list,
    lowercase_cols: list,
    outlier_numeric_cols: list,
    irrelevant_cols: list,
    dtype_mappings: dict,
    # Feature engineering configs
    tenure_bins: list,
    tenure_labels: list,
    totalcharges_bins: list,
    totalcharges_labels: list,
    interaction_pairs: list,
    ratio_pairs: list,
    categorical_cols_list: list,
    target_col: str
) -> tuple[pd.DataFrame, object]:
    """
    ZenML step: preprocess raw df into processed DataFrame ready for splitting:
    - Convert object-to-numeric
    - Encode binary columns
    - Handle missing, duplicates, formats, outliers
    - Drop irrelevant features, convert dtypes
    - Feature engineering: binning, interactions, ratios
    - Drop high-cardinality categorical
    - Scale & encode (ColumnTransformer), returns processed df with only numeric columns (features) plus target_col
      and returns fitted transformer for inference.
    """
    # 1. Convert object columns that represent numeric values (e.g., 'TotalCharges')
    df1 = convert_object_to_numeric(df, columns=object_numeric_cols)

    # 2. Encode binary columns (Yes/No etc.) according to mapping, including 'Churn'
    df2 = encode_binary_columns(df1, mapping=binary_mapping)

    # 3. Handle missing values (e.g., after conversion some NaN may appear)
    df3 = handle_missing_values(df2, numeric_cols=missing_numeric_cols, categorical_cols=missing_categorical_cols)

    # 4. Drop duplicates
    df4 = handle_duplicates(df3, subset=drop_duplicate_subset)

    # 5. Standardize formats (none for Telco, but placeholder)
    df5 = standardize_formats(df4, datetime_cols=datetime_cols, lowercase_cols=lowercase_cols)

    # 6. Handle outliers on numeric columns
    df6 = handle_outliers(df5, numeric_cols=outlier_numeric_cols)

    # 7. Drop irrelevant features (e.g., 'customerID')
    df7 = drop_irrelevant_features(df6, irrelevant_cols)

    # 8. Convert explicit dtypes if needed
    df8 = convert_dtypes(df7, dtype_mappings)

    # 9. Feature engineering: bin tenure
    df9 = bin_numerical_feature(df8, column='tenure', bins=tenure_bins, labels=tenure_labels)
    # 10. Feature engineering: bin TotalCharges
    df10 = bin_numerical_feature(df9, column='TotalCharges', bins=totalcharges_bins, labels=totalcharges_labels)
    # 11. Interaction features
    df11 = create_interaction_features(df10, feature_pairs=interaction_pairs)
    # 12. Ratio features
    df12 = create_ratio_features(df11, ratio_pairs=ratio_pairs)

    # 13. Drop high-cardinality categorical columns
    df13, retained_categoricals = drop_high_cardinality_categorical(df12, categorical_cols=categorical_cols_list)

    # 14. Prepare lists for scaling & encoding:
    #    Numeric columns: all numeric dtypes except target_col
    numeric_cols = [col for col in df13.columns
                    if pd.api.types.is_numeric_dtype(df13[col]) and col != target_col]
    #    Categorical columns: retained_categoricals plus any newly generated binned columns
    categorical_cols = [col for col in retained_categoricals
                        if col in df13.columns]
    # Note: Binned columns 'tenure_binned' and 'TotalCharges_binned' should be included in categorical_cols_list
    # and thus in retained_categoricals if not dropped.

    # 15. Scale and encode
    features_df = df13.drop(columns=[target_col])
    df_transformed, transformer = scale_and_encode(
        features_df,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        scaler_type='standard'
    )
    # 16. Reattach target_col
    df_transformed[target_col] = df13[target_col].values

    # 17. Save transformer for inference
    os.makedirs("model", exist_ok=True)
    transformer_path = os.path.join("model", "transformer.pkl")
    joblib.dump(transformer, transformer_path)
    logger.info(f"Saved preprocessing transformer to {transformer_path}")
    with mlflow.start_run(run_name="transformer_logging", nested=True):
        mlflow.log_artifact(transformer_path, artifact_path="transformer")

    return df_transformed, transformer

# -----------------------
# Step 3: Splitting
@step
def splitting_step(df: pd.DataFrame, target_col: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    ZenML step: split processed df into train/val/test.
    """
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df,
        target_col=target_col,
        test_size=0.2,
        val_size=0.1,
        random_state=42,
        output_dir="data/processed"
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

# -----------------------
# Step 4: Training 
@step
def training_step(X_train: pd.DataFrame,
                  y_train: pd.Series,
                  X_val: pd.DataFrame,
                  y_val: pd.Series,
                  resample_method: str,
                  use_class_weight: bool,
                  n_iter: int = 10,
                  cv: int = 3) -> dict:
    """
    ZenML step: train multiple candidate models, optionally resample training set.
    - resample_method: 'none', 'smote', or 'undersample'
    - use_class_weight: True/False
    """
    return train_and_log_all_models(
        X_train, y_train, X_val, y_val,
        n_iter=n_iter,
        cv=cv,
        resample_method=resample_method,
        use_class_weight=use_class_weight
    )

# -----------------------
# Step 5: Evaluation
@step
def evaluation_step(model_infos: dict, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    """
    ZenML step: evaluate all models on test set, select and register best, save plots under artifacts/evaluation.
    """
    os.makedirs("artifacts/evaluation", exist_ok=True)
    return evaluate_all_models_on_test(model_infos, X_test, y_test, output_dir="artifacts/evaluation")

# -----------------------
# Pipeline definition
@pipeline(enable_cache=False)
def full_pipeline(
    raw_data_path: str,
    # Preprocessing configs
    object_numeric_cols: list,
    binary_mapping: dict,
    missing_numeric_cols: list,
    missing_categorical_cols: list,
    drop_duplicate_subset: list,
    datetime_cols: list,
    lowercase_cols: list,
    outlier_numeric_cols: list,
    irrelevant_cols: list,
    dtype_mappings: dict,
    # Feature engineering configs
    tenure_bins: list,
    tenure_labels: list,
    totalcharges_bins: list,
    totalcharges_labels: list,
    interaction_pairs: list,
    ratio_pairs: list,
    categorical_cols_list: list,
    target_col: str,
    # Training configs
    resample_method: str,
    use_class_weight: bool,
    n_iter: int,
    cv: int
):
    """
    ZenML pipeline that performs end-to-end steps including automated deployment.
    """
    # 1. Ingest
    df = ingest_step(raw_data_path=raw_data_path)
    # 2. Preprocess & feature engineering
    df_processed, transformer = preprocessing_step(
        df,
        object_numeric_cols=object_numeric_cols,
        binary_mapping=binary_mapping,
        missing_numeric_cols=missing_numeric_cols,
        missing_categorical_cols=missing_categorical_cols,
        drop_duplicate_subset=drop_duplicate_subset,
        datetime_cols=datetime_cols,
        lowercase_cols=lowercase_cols,
        outlier_numeric_cols=outlier_numeric_cols,
        irrelevant_cols=irrelevant_cols,
        dtype_mappings=dtype_mappings,
        tenure_bins=tenure_bins,
        tenure_labels=tenure_labels,
        totalcharges_bins=totalcharges_bins,
        totalcharges_labels=totalcharges_labels,
        interaction_pairs=interaction_pairs,
        ratio_pairs=ratio_pairs,
        categorical_cols_list=categorical_cols_list,
        target_col=target_col
    )
    # 3. Split
    X_train, X_val, X_test, y_train, y_val, y_test = splitting_step(df_processed, target_col=target_col)
    # 4. Train & select best
    model_infos = training_step(
        X_train, y_train, X_val, y_val,
        resample_method=resample_method,
        use_class_weight=use_class_weight,
        n_iter=n_iter,
        cv=cv
    )
    # 5. Evaluate
    evaluation_results = evaluation_step(model_infos, X_test, y_test)
    logger.info(f"Evaluation metrics: {evaluation_results}")

    logger.info("Full pipeline completed successfully.")
