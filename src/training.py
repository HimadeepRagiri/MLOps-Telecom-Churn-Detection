import os
import joblib
import numpy as np
import mlflow
import mlflow.sklearn
from loguru import logger
import subprocess
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score

# For balancing
from src.preprocessing import balance_data

def get_model_candidates(use_class_weight: bool = True):
    """
    Returns a dict: model_name -> (estimator, hyperparam distributions).
    If use_class_weight=True, set class_weight='balanced' for models that support it.
    """
    candidates = {}
    # Logistic Regression
    if use_class_weight:
        lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
    else:
        lr = LogisticRegression(max_iter=1000, random_state=42)
    lr_params = {
        'C': np.logspace(-4, 4, 20),
        'penalty': ['l2'],
        'solver': ['lbfgs']
    }
    candidates['logistic_regression'] = (lr, lr_params)

    # Random Forest
    if use_class_weight:
        rf = RandomForestClassifier(random_state=42, class_weight='balanced')
    else:
        rf = RandomForestClassifier(random_state=42)
    rf_params = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 5, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    candidates['random_forest'] = (rf, rf_params)

    # XGBoost (XGBoost uses scale_pos_weight instead of class_weight)
    xgb = XGBClassifier(eval_metric='logloss', random_state=42)
    xgb_params = {
        'n_estimators': [100, 200, 500],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.6, 0.8, 1.0],
        'scale_pos_weight': [2.5, 2.7, 3.0]  
    }
    candidates['xgboost'] = (xgb, xgb_params)
    return candidates

def train_and_log_all_models(X_train, y_train, X_val, y_val,
                          n_iter: int = 10,
                          cv: int = 3,
                          resample_method: str = 'none',
                          use_class_weight: bool = True):
    """
    Train multiple models with RandomizedSearchCV, track experiments in MLflow,
    optionally apply resampling to X_train/y_train, and select best on validation ROC AUC.
    - resample_method: 'none', 'smote', or 'undersample' (uses balance_data).
    - use_class_weight: if True, sets class_weight='balanced' for models that support it.
    Returns: dict of all models and their info.
    """
    # Optionally balance training data
    X_train_bal, y_train_bal = balance_data(X_train, y_train, method=resample_method, random_state=42)
    if resample_method != 'none':
        logger.info(f"Training data balanced with method '{resample_method}': original n={len(y_train)}, balanced n={len(y_train_bal)}")
    else:
        X_train_bal, y_train_bal = X_train, y_train

    mlflow.set_experiment("Churn_detection_experiment")
    candidates = get_model_candidates(use_class_weight=use_class_weight)
    model_infos = {}

    for model_name, (estimator, param_dist) in candidates.items():
        logger.info(f"Training candidate model: {model_name}")
        search = RandomizedSearchCV(estimator, param_distributions=param_dist,
                                    n_iter=n_iter, scoring='roc_auc',
                                    cv=cv, random_state=42, n_jobs=2)
        with mlflow.start_run(run_name=model_name) as run:
            # Fit on (possibly) balanced data
            search.fit(X_train_bal, y_train_bal)
            best_est = search.best_estimator_
            best_p = search.best_params_
            # Evaluate on validation set
            try:
                y_val_pred_proba = best_est.predict_proba(X_val)[:, 1]
            except AttributeError:
                # Some models (if custom) may not have predict_proba
                y_val_pred_proba = best_est.decision_function(X_val)
                from sklearn.preprocessing import MinMaxScaler
                y_val_pred_proba = MinMaxScaler().fit_transform(y_val_pred_proba.reshape(-1, 1)).flatten()
            val_score = roc_auc_score(y_val, y_val_pred_proba)
            logger.info(f"Validation ROC AUC for '{model_name}': {val_score:.4f}")

            # Log to MLflow
            mlflow.log_param("model_name", model_name)
            mlflow.log_param("resample_method", resample_method)
            mlflow.log_param("use_class_weight", use_class_weight)
            for param, val in best_p.items():
                mlflow.log_param(param, val)
            mlflow.log_metric("val_roc_auc", val_score)
            
            # Log model artifact
            mlflow.sklearn.log_model(best_est, artifact_path="model")

            # MLflow Code Version Tracking
            try:
                git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
                mlflow.log_param("git_commit", git_commit)
            except Exception:
                mlflow.log_param("git_commit", "unknown")

            # Save model locally
            os.makedirs("model", exist_ok=True)
            model_path = os.path.join("model", f"{model_name}.pkl")
            joblib.dump(best_est, model_path)
            logger.info(f"Saved {model_name} to {model_path}")

            model_infos[model_name] = {
                "model": best_est,
                "params": best_p,
                "val_score": val_score,
                "run_id": run.info.run_id,
                "model_path": model_path
            }

        logger.info(f"Finished training '{model_name}'.")

    return  model_infos
