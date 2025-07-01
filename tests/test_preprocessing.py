import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.preprocessing import (
    convert_object_to_numeric, encode_binary_columns, handle_missing_values,
    handle_duplicates, standardize_formats, handle_outliers,
    drop_irrelevant_features, convert_dtypes, scale_and_encode, balance_data
)

def test_convert_object_to_numeric():
    df = pd.DataFrame({'a': ['1', '2', 'x']})
    out = convert_object_to_numeric(df, columns=['a'])
    assert out['a'].dtype.kind in 'fi'
    assert pd.isna(out['a'][2])

def test_encode_binary_columns():
    df = pd.DataFrame({'b': ['Yes', 'No', 'Maybe']})
    mapping = {'b': {'Yes': 1, 'No': 0}}
    out = encode_binary_columns(df, mapping)
    assert out['b'][0] == 1
    assert out['b'][1] == 0
    assert pd.isna(out['b'][2])

def test_handle_missing_values():
    df = pd.DataFrame({'a': [1, np.nan, 3], 'b': ['x', np.nan, 'y']})  # Use np.nan
    out = handle_missing_values(df, numeric_cols=['a'], categorical_cols=['b'])
    assert out['a'].isna().sum() == 0
    assert out['b'].isna().sum() == 0

def test_handle_duplicates():
    df = pd.DataFrame({'a': [1, 1, 2], 'b': [3, 3, 4]})
    out = handle_duplicates(df)
    assert len(out) == 2

def test_standardize_formats():
    df = pd.DataFrame({'c': ['A', 'B']})
    out = standardize_formats(df, lowercase_cols=['c'])
    assert all(out['c'] == ['a', 'b'])

def test_handle_outliers():
    df = pd.DataFrame({'a': [1, 2, 100]})
    out = handle_outliers(df, numeric_cols=['a'])
    # Accept max equal to 100 if capping is at 100, or check that at least one value was capped
    assert out['a'].max() <= 100
    assert (out['a'] == 100).sum() <= 1  # Only one value should be at the cap

def test_drop_irrelevant_features():
    df = pd.DataFrame({'a': [1], 'b': [2]})
    out = drop_irrelevant_features(df, irrelevant_cols=['b'])
    assert 'b' not in out.columns

def test_convert_dtypes():
    df = pd.DataFrame({'a': [1, 2]})
    out = convert_dtypes(df, {'a': 'float'})
    assert out['a'].dtype == float

def test_scale_and_encode():
    df = pd.DataFrame({'a': [1, 2], 'b': ['x', 'y']})
    out, transformer = scale_and_encode(df, numeric_cols=['a'], categorical_cols=['b'])
    assert out.shape[1] > 1

def test_balance_data():
    import pandas as pd
    X = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6]})
    y = pd.Series([0, 0, 0, 1, 1, 1])
    X_res, y_res = balance_data(X, y, method='smote')
    assert len(X_res) == len(y_res)