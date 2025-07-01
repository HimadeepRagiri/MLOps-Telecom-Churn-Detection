import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
from src.feature_engineering import (
    create_ratio_features, create_interaction_features, bin_numerical_feature, drop_high_cardinality_categorical
)

def test_create_ratio_features():
    df = pd.DataFrame({'x': [2, 4], 'y': [1, 2]})
    out = create_ratio_features(df, [('x', 'y', 'x_over_y')])
    assert np.allclose(out['x_over_y'], [2, 2])  # Use allclose for floats

def test_create_interaction_features():
    df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    out = create_interaction_features(df, [('a', 'b', 'a_times_b')])
    assert all(out['a_times_b'] == [3, 8])

def test_bin_numerical_feature():
    df = pd.DataFrame({'val': [5, 15, 25]})
    out = bin_numerical_feature(df, 'val', bins=[0, 10, 20, 30], labels=['low', 'mid', 'high'])
    assert set(out['val_binned']) <= set(['low', 'mid', 'high'])

def test_drop_high_cardinality_categorical():
    df = pd.DataFrame({'cat': ['a']*10 + ['b']*41})
    out, retained = drop_high_cardinality_categorical(df, ['cat'], threshold=20)
    assert 'cat' not in out.columns or 'cat' in retained