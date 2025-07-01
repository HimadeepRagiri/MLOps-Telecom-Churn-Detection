import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
from src.training import train_and_log_all_models

def test_train_and_log_all_models_runs(tmp_path):
    # Minimal data for smoke test
    X_train = pd.DataFrame({'a': [1, 2, 3, 4], 'b': [0, 1, 0, 1]})
    y_train = pd.Series([0, 1, 0, 1])
    X_val = pd.DataFrame({'a': [5, 6], 'b': [1, 0]})
    y_val = pd.Series([1, 0])
    model_infos = train_and_log_all_models(X_train, y_train, X_val, y_val, n_iter=1, cv=2, resample_method='none', use_class_weight=True)
    assert isinstance(model_infos, dict)
    assert len(model_infos) > 0