import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from loguru import logger

# For balancing
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

def convert_object_to_numeric(df: pd.DataFrame, columns: list, errors: str = 'coerce') -> pd.DataFrame:
    """
    Convert object/string columns that represent numeric values into numeric dtype.
    - columns: list of column names to convert via pd.to_numeric.
    - errors: 'coerce' (invalid parsing set as NaN) or 'raise'.
    """
    df = df.copy()
    for col in columns:
        if col in df.columns:
            before_non_numeric = df[col].dtype
            df[col] = pd.to_numeric(df[col], errors=errors)
            logger.info(f"Converted column '{col}' from {before_non_numeric} to {df[col].dtype} with errors='{errors}'")
        else:
            logger.warning(f"Column '{col}' not found for numeric conversion")
    return df

def encode_binary_columns(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    """
    Encode binary categorical columns to 0/1 according to mapping.
    - mapping: dict of form { 'col_name': {positive_label:1, negative_label:0, ...} }
      E.g., {'Churn': {'Yes':1, 'No':0}, 'Partner': {'Yes':1, 'No':0}}
    - If some values not in mapping, they become NaN; handle after.
    """
    df = df.copy()
    for col, map_dict in mapping.items():
        if col in df.columns:
            df[col] = df[col].map(map_dict)
            num_unmapped = df[col].isna().sum()
            if num_unmapped > 0:
                logger.warning(f"Column '{col}': {num_unmapped} values could not be mapped to binary; they are NaN now")
            else:
                logger.info(f"Encoded binary column '{col}' with mapping {map_dict}")
        else:
            logger.warning(f"Column '{col}' not found for binary encoding")
    return df

def handle_missing_values(df: pd.DataFrame,
                          strategy_numeric: str = 'median',
                          strategy_categorical: str = 'most_frequent',
                          numeric_cols: list = None,
                          categorical_cols: list = None) -> pd.DataFrame:
    """
    Impute missing values:
    - numeric_cols: list of numeric columns to impute (SimpleImputer).
    - categorical_cols: list of categorical columns to impute.
    """
    df = df.copy()
    if numeric_cols:
        num_imputer = SimpleImputer(strategy=strategy_numeric)
        df[numeric_cols] = num_imputer.fit_transform(df[numeric_cols])
        logger.info(f"Imputed missing numeric columns {numeric_cols} using strategy='{strategy_numeric}'")
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy=strategy_categorical)
        df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])
        logger.info(f"Imputed missing categorical columns {categorical_cols} using strategy='{strategy_categorical}'")
    return df

def handle_duplicates(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """
    Drop duplicate rows. If subset provided, consider only those columns to identify duplicates.
    """
    df = df.copy()
    before = len(df)
    if subset is not None and len(subset) == 0:
        subset = None
    df = df.drop_duplicates(subset=subset)
    after = len(df)
    logger.info(f"Dropped {before - after} duplicate rows using subset={subset}")
    return df

def standardize_formats(df: pd.DataFrame, datetime_cols: list = None, lowercase_cols: list = None) -> pd.DataFrame:
    """
    Standardize formats:
    - datetime_cols: convert to datetime dtype (errors='coerce').
    - lowercase_cols: convert string columns to lowercase.
    """
    df = df.copy()
    if datetime_cols:
        for col in datetime_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                logger.info(f"Standardized datetime column '{col}'")
            else:
                logger.warning(f"Column '{col}' not found for datetime standardization")
    if lowercase_cols:
        for col in lowercase_cols:
            if col in df.columns:
                df[col] = df[col].astype(str).str.lower()
                logger.info(f"Lowercased strings in column '{col}'")
            else:
                logger.warning(f"Column '{col}' not found for lowercase conversion")
    return df

def handle_outliers(df: pd.DataFrame, numeric_cols: list, method: str = 'iqr', factor: float = 1.5) -> pd.DataFrame:
    """
    Handle outliers in numeric columns:
    - method 'iqr': cap values outside [Q1 - factor*IQR, Q3 + factor*IQR].
    """
    df = df.copy()
    if method == 'iqr':
        for col in numeric_cols:
            if col not in df.columns:
                logger.warning(f"Column '{col}' not found for outlier handling")
                continue
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - factor * IQR
            upper = Q3 + factor * IQR
            before_extremes = ((df[col] < lower) | (df[col] > upper)).sum()
            df[col] = df[col].clip(lower=lower, upper=upper)
            logger.info(f"Capped outliers in '{col}': {before_extremes} values clipped to [{lower}, {upper}]")
    else:
        logger.warning(f"Unknown outlier handling method '{method}'; no action taken")
    return df

def drop_irrelevant_features(df: pd.DataFrame, irrelevant_cols: list) -> pd.DataFrame:
    """
    Drop columns that are irrelevant (e.g., customerID). Irrelevant columns list supplied after EDA.
    """
    df = df.copy()
    df = df.drop(columns=irrelevant_cols, errors='ignore')
    logger.info(f"Dropped irrelevant columns: {irrelevant_cols}")
    return df

def convert_dtypes(df: pd.DataFrame, dtype_mappings: dict) -> pd.DataFrame:
    """
    Convert columns to specified dtypes: dtype_mappings = {col_name: dtype_str}.
    """
    df = df.copy()
    for col, dtype in dtype_mappings.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
                logger.info(f"Converted '{col}' to dtype {dtype}")
            except Exception as e:
                logger.warning(f"Failed to convert '{col}' to {dtype}: {e}")
        else:
            logger.warning(f"Column '{col}' not found for dtype conversion")
    return df

def scale_and_encode(df: pd.DataFrame,
                     numeric_cols: list,
                     categorical_cols: list,
                     scaler_type: str = 'standard') -> tuple[pd.DataFrame, object]:
    """
    Scale numeric columns and one-hot encode categorical columns via ColumnTransformer.
    Returns transformed DataFrame and fitted ColumnTransformer.
    - numeric_cols: list of column names (after any binary encoding or binning).
    - categorical_cols: list of column names to one-hot encode.
    - scaler_type: 'standard' or 'minmax'.
    """
    df = df.copy()
    transformers = []
    if numeric_cols:
        if scaler_type == 'standard':
            scaler = StandardScaler()
        else:
            scaler = MinMaxScaler()
        transformers.append(('num', scaler, numeric_cols))
    if categorical_cols:
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        transformers.append(('cat', encoder, categorical_cols))
    if transformers:
        col_transformer = ColumnTransformer(transformers=transformers, remainder='drop')
        arr = col_transformer.fit_transform(df)
        # Build feature names
        feature_names = []
        if numeric_cols:
            feature_names.extend(numeric_cols)
        if categorical_cols:
            cat_features = col_transformer.named_transformers_['cat'].get_feature_names_out(categorical_cols)
            feature_names.extend(cat_features.tolist())
        df_transformed = pd.DataFrame(arr, columns=feature_names)
        logger.info(f"Scaled numeric columns {numeric_cols} and encoded categorical columns {categorical_cols}")
        return df_transformed, col_transformer
    else:
        logger.warning("No numeric or categorical columns provided to scale_and_encode; returning original df")
        return df, None

def balance_data(X: pd.DataFrame, y: pd.Series, method: str = 'none', random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """
    Balance training data:
    - method: 'none' (no resampling), 'smote' (oversample minority), 'undersample' (random undersampling majority).
    Returns balanced X_res, y_res.
    """
    if method not in ('none', 'smote', 'undersample'):
        logger.warning(f"Unknown balance method '{method}'; proceeding without resampling")
        return X, y
    if method == 'none':
        return X, y
    if method == 'smote':
        try:
            smote = SMOTE(random_state=random_state)
            X_res, y_res = smote.fit_resample(X, y)
            logger.info(f"Applied SMOTE: before n={len(y)}, after n={len(y_res)}")
            return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)
        except Exception as e:
            logger.error(f"SMOTE failed: {e}. Returning original data")
            return X, y
    if method == 'undersample':
        try:
            rus = RandomUnderSampler(random_state=random_state)
            X_res, y_res = rus.fit_resample(X, y)
            logger.info(f"Applied RandomUnderSampler: before n={len(y)}, after n={len(y_res)}")
            return pd.DataFrame(X_res, columns=X.columns), pd.Series(y_res)
        except Exception as e:
            logger.error(f"RandomUnderSampler failed: {e}. Returning original data")
            return X, y
