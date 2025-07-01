import pandas as pd
from loguru import logger

def create_ratio_features(df: pd.DataFrame, ratio_pairs: list) -> pd.DataFrame:
    """
    Create ratio features generically.
    - ratio_pairs: list of tuples (numerator_col, denominator_col, new_col_name).
    """
    df = df.copy()
    for num_col, den_col, new_col in ratio_pairs:
        if num_col in df.columns and den_col in df.columns:
            epsilon = 1e-6
            df[new_col] = df[num_col] / (df[den_col] + epsilon)
            logger.info(f"Created ratio feature '{new_col}' = {num_col}/{den_col}")
        else:
            logger.warning(f"Cannot create ratio '{new_col}': columns '{num_col}' or '{den_col}' not found")
    return df

def create_interaction_features(df: pd.DataFrame, feature_pairs: list) -> pd.DataFrame:
    """
    Create interaction features generically.
    - feature_pairs: list of tuples (col1, col2, new_col_name).
    """
    df = df.copy()
    for col1, col2, new_col in feature_pairs:
        if col1 in df.columns and col2 in df.columns:
            df[new_col] = df[col1] * df[col2]
            logger.info(f"Created interaction feature '{new_col}' = {col1} * {col2}")
        else:
            logger.warning(f"Cannot create interaction '{new_col}': columns '{col1}' or '{col2}' not found")
    return df

def create_custom_datetime_features(df: pd.DataFrame, datetime_col: str) -> pd.DataFrame:
    """
    Extract datetime parts generically.
    - datetime_col: name of datetime column.
    """
    df = df.copy()
    if datetime_col and datetime_col in df.columns and pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[f"{datetime_col}_year"] = df[datetime_col].dt.year
        df[f"{datetime_col}_month"] = df[datetime_col].dt.month
        df[f"{datetime_col}_day"] = df[datetime_col].dt.day
        df[f"{datetime_col}_weekday"] = df[datetime_col].dt.weekday
        df[f"{datetime_col}_hour"] = df[datetime_col].dt.hour
        logger.info(f"Extracted datetime features from '{datetime_col}'")
    else:
        if datetime_col:
            logger.warning(f"Column '{datetime_col}' not found or not datetime; skipping datetime feature extraction")
    return df

def drop_high_cardinality_categorical(df: pd.DataFrame, categorical_cols: list, threshold: int = 50) -> tuple[pd.DataFrame, list]:
    """
    Drop categorical columns with unique values above threshold.
    - categorical_cols: list of categorical column names to check.
    Returns updated df and retained categorical columns list.
    """
    df = df.copy()
    retained = []
    for col in categorical_cols:
        if col in df.columns:
            nuniques = df[col].nunique()
            if nuniques <= threshold:
                retained.append(col)
            else:
                df.drop(columns=[col], inplace=True)
                logger.info(f"Dropped high-cardinality column '{col}' ({nuniques} unique values)")
        else:
            logger.warning(f"Column '{col}' not found for high-cardinality check")
    return df, retained

def bin_numerical_feature(df: pd.DataFrame, column: str, bins: list, labels: list = None, right: bool = False) -> pd.DataFrame:
    """
    Bin a numerical feature into categories; same as in preprocessing.
    - column: name of numeric column.
    - bins: list of edges.
    - labels: list of labels for bins.
    Returns df with new column f"{column}_binned".
    """
    df = df.copy()
    if column not in df.columns:
        logger.warning(f"Column '{column}' not found for binning in feature_engineering")
        return df
    try:
        binned = pd.cut(df[column], bins=bins, labels=labels, right=right)
        new_col = f"{column}_binned"
        df[new_col] = binned.astype(str)
        logger.info(f"Binned feature '{column}' into '{new_col}'")
    except Exception as e:
        logger.error(f"Error binning feature '{column}': {e}")
    return df
