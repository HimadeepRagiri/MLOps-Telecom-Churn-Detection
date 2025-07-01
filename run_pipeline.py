import pandas as pd
from src.data_ingestion import load_raw_data
from pipelines.training_pipeline import full_pipeline

def compute_bins_from_series(series: pd.Series, n_bins: int = 4):
    """
    Compute bin edges based on quantiles for series (numeric, no NaN).
    Returns bins list (length n_bins+1) and labels.
    """
    # Drop NaN
    clean = series.dropna()
    # Compute quantiles at equal intervals
    quantiles = [clean.quantile(q) for q in [i / n_bins for i in range(n_bins + 1)]]
    # Ensure strictly increasing bins: e.g., if duplicates, adjust slightly
    bins = []
    prev = None
    for q in quantiles:
        if prev is None:
            bins.append(q)
            prev = q
        else:
            # If q <= prev due to duplicates, add a tiny epsilon
            if q <= prev:
                q = prev + 1e-4
            bins.append(q)
            prev = q
    # Labels: generic 'bin0', 'bin1', ...
    labels = [f"bin{i}" for i in range(len(bins) - 1)]
    return bins, labels

if __name__ == "__main__":
    # 1. Paths
    raw_data_path = "data/raw/Telco-Customer-Churn.csv"  

    # 2. Load raw data to compute dynamic configs (e.g., bins)
    df_raw = load_raw_data(raw_data_path)

    # 3. Configuration based on EDA
    # 3a. Convert object columns representing numeric
    object_numeric_cols = ["TotalCharges"]  # convert to numeric

    # 3b. Binary mapping for Yes/No and other binary-like columns
    #    For columns with 'No internet service' or 'No phone service', map to 0.
    binary_mapping = {
        'Churn': {'Yes': 1, 'No': 0},
        'Partner': {'Yes': 1, 'No': 0},
        'Dependents': {'Yes': 1, 'No': 0},
        'PhoneService': {'Yes': 1, 'No': 0},
        'PaperlessBilling': {'Yes': 1, 'No': 0},
        'MultipleLines': {'Yes': 1, 'No': 0, 'No phone service': 0},
        'OnlineSecurity': {'Yes': 1, 'No': 0, 'No internet service': 0},
        'OnlineBackup': {'Yes': 1, 'No': 0, 'No internet service': 0},
        'DeviceProtection': {'Yes': 1, 'No': 0, 'No internet service': 0},
        'TechSupport': {'Yes': 1, 'No': 0, 'No internet service': 0},
        'StreamingTV': {'Yes': 1, 'No': 0, 'No internet service': 0},
        'StreamingMovies': {'Yes': 1, 'No': 0, 'No internet service': 0},
        'gender': {'Male': 1, 'Female': 0}
    }

    # 3c. Missing value lists: after conversion, check if any missing appear
    #     For Telco, EDA showed no missing initially; after conversion, some TotalCharges may become NaN if blank.
    #     Let's detect:
    df_tmp = df_raw.copy()
    # Convert TotalCharges to numeric to see missing
    df_tmp['TotalCharges'] = pd.to_numeric(df_tmp['TotalCharges'], errors='coerce')
    missing_numeric_cols = [col for col in ['MonthlyCharges', 'TotalCharges'] if df_tmp[col].isna().sum() > 0]
    # If none have missing, still include TotalCharges to impute zero or median
    if 'TotalCharges' not in missing_numeric_cols:
        missing_numeric_cols.append('TotalCharges')
    missing_categorical_cols = []  # EDA showed no missing categorical

    # 3d. Duplicates: none
    drop_duplicate_subset = []  

    # 3e. No datetime columns in Telco
    datetime_cols = []
    datetime_feature_col = []

    # 3f. Lowercase columns: none
    lowercase_cols = []

    # 3g. Outlier numeric columns: monthly charges, tenure, TotalCharges
    outlier_numeric_cols = ['MonthlyCharges', 'tenure', 'TotalCharges']

    # 3h. Irrelevant columns: customerID
    irrelevant_cols = ['customerID']

    # 3i. Dtype mappings: none extra after conversion
    dtype_mappings = {}

    # 3j. Feature engineering: compute bins for tenure and TotalCharges
    # Tenure is integer 0-72; we can set fixed bins as per EDA:
    tenure_bins = [int(0), int(12), int(24), int(48), int(df_raw['tenure'].max())]
    tenure_labels = ['0-12', '13-24', '25-48', '49+']

    # TotalCharges: dynamic bins based on quantiles
    # First ensure conversion to numeric
    tc_series = pd.to_numeric(df_raw['TotalCharges'], errors='coerce').dropna()
    if len(tc_series) > 0:
        totalcharges_bins, totalcharges_labels = compute_bins_from_series(tc_series, n_bins=4)
        # Convert to Python float and str
        totalcharges_bins = [float(x) for x in totalcharges_bins]
        totalcharges_labels = [str(x) for x in totalcharges_labels]
    else:
        # Fallback: arbitrary bins
        totalcharges_bins = [0.0, 400.0, 1400.0, 2800.0, 9000.0]
        totalcharges_labels = ['0-399', '400-1399', '1400-2799', '2800+']

    # 3k. Interaction & ratio pairs: as determined by EDA
    # interaction between MonthlyCharges and tenure
    interaction_pairs = [('MonthlyCharges', 'tenure', 'Monthly_x_tenure')]
    ratio_pairs = []  # likely not needed for Telco churn; keep empty 

    # 3l. Categorical columns list (before encoding) that we will one-hot encode AFTER drop_high_cardinality:
    # These include multi-category features:
    categorical_cols_list = [
        'InternetService', 'Contract', 'PaymentMethod',
        'tenure_binned', 'TotalCharges_binned'
    ]

    # 3m. Target column
    target_col = 'Churn'

    # 3n. Training configs
    resample_method = 'smote'  # to handle imbalance
    use_class_weight = True
    n_iter = 10  # or larger for more thorough search
    cv = 3

    # 4. Run ZenML pipeline
    full_pipeline(
        raw_data_path=raw_data_path,
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
        target_col=target_col,
        resample_method=resample_method,
        use_class_weight=use_class_weight,
        n_iter=n_iter,
        cv=cv
    )

    # 5. After pipeline run: ZenML steps have:
    #    - Saved transformer at "model/transformer.pkl"
    #    - Saved best model at "model/<model_name>.pkl"
    #    - Saved processed train/val/test splits under data/processed/
    #    - Saved evaluation plots under artifacts/evaluation/

    print("Pipeline execution finished.")
