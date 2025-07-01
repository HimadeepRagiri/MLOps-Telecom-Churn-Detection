import pandas as pd
from loguru import logger

def inspect_dataframe(df: pd.DataFrame, max_rows: int = 5):
    """
    Print basic info: head, tail, info, dtypes, missing count, basic stats.
    """
    logger.info("DataFrame head:")
    display_head = df.head(max_rows)
    print(display_head)
    logger.info("DataFrame tail:")
    print(df.tail(max_rows))
    logger.info("DataFrame info:")
    print(df.info())  # prints to stdout
    logger.info("Dtypes and missing values count:")
    missing = df.isna().sum()
    dtypes = df.dtypes
    summary = pd.DataFrame({"dtype": dtypes, "missing_count": missing, "missing_pct": missing / len(df) * 100})
    print(summary)
    logger.info("Basic descriptive statistics for numeric columns:")
    print(df.describe(include='number').T)
    duplicate_count = df.duplicated().sum()
    duplicate_pct = 100 * duplicate_count / len(df)
    logger.info(f"Duplicate rows: {duplicate_count} ({duplicate_pct:.2f}%)")
    print(f"Duplicate rows: {duplicate_count} ({duplicate_pct:.2f}%)")
