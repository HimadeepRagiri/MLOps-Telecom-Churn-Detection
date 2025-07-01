import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score
from loguru import logger
import os
import joblib
import subprocess
import mlflow
from mlflow.tracking import MlflowClient

def evaluate_all_models_on_test(model_infos: dict, X_test, y_test, output_dir: str = None):
    """
    Evaluate all models on test set, log metrics, return best model info.
    Also plots and optionally saves:
      - ROC curve
      - Precision-Recall curve
      - Confusion matrix
      - Classification report (printed and returned)
    """
    best_model_name = None
    best_model = None
    best_score = -float("inf")
    best_run_id = None
    best_model_path = None

    for model_name, info in model_infos.items():
        model = info["model"]
        try:
            y_proba = model.predict_proba(X_test)[:, 1]
        except AttributeError:
            y_proba = model.decision_function(X_test)
            from sklearn.preprocessing import MinMaxScaler
            y_proba = MinMaxScaler().fit_transform(y_proba.reshape(-1, 1)).flatten()
            logger.info("Used decision_function and scaled to [0,1] for evaluation")

        y_pred = (y_proba >= 0.5).astype(int)

        # Classification report
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        logger.info(f"Classification report for {model_name}:\n{classification_report(y_test, y_pred)}")

        # ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        roc_curve_path = None
        if output_dir:
            roc_curve_path = f"{output_dir}/roc_curve_{model_name}.png"
            plt.savefig(roc_curve_path)
        plt.close()

        # Precision-Recall curve
        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        ap_score = average_precision_score(y_test, y_proba)
        plt.figure()
        plt.plot(recall, precision, label=f'Precision-Recall curve (AP = {ap_score:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="upper right")
        pr_curve_path = None
        if output_dir:
            pr_curve_path = f"{output_dir}/precision_recall_curve_{model_name}.png"
            plt.savefig(pr_curve_path)
        plt.close()
        logger.info(f"Average precision (AP) score for {model_name}: {ap_score:.4f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title(f'Confusion Matrix - {model_name}')
        cm_path = None
        if output_dir:
            cm_path = f"{output_dir}/confusion_matrix_{model_name}.png"
            plt.savefig(cm_path)
        plt.close()

        # Log metrics and artifacts to MLflow 
        with mlflow.start_run(run_id=info["run_id"]):
            mlflow.log_metric("test_roc_auc", roc_auc)
            mlflow.log_metric("test_average_precision", ap_score)
            mlflow.log_metric("test_TN", cm[0][0])
            mlflow.log_metric("test_FP", cm[0][1])
            mlflow.log_metric("test_FN", cm[1][0])
            mlflow.log_metric("test_TP", cm[1][1])
            # Log plots if available
            if roc_curve_path and os.path.exists(roc_curve_path):
                mlflow.log_artifact(roc_curve_path, artifact_path="plots")
            if pr_curve_path and os.path.exists(pr_curve_path):
                mlflow.log_artifact(pr_curve_path, artifact_path="plots")
            if cm_path and os.path.exists(cm_path):
                mlflow.log_artifact(cm_path, artifact_path="plots")
            # Log classification report as text
            import json
            report_txt_path = None
            if output_dir:
                report_txt_path = f"{output_dir}/classification_report_{model_name}.txt"
                with open(report_txt_path, "w") as f:
                    f.write(json.dumps(report_dict, indent=2))
                mlflow.log_artifact(report_txt_path, artifact_path="reports")
            # MLflow Code Version Tracking
            try:
                git_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
                mlflow.log_param("git_commit", git_commit)
            except Exception:
                mlflow.log_param("git_commit", "unknown")

        # Select best model based on test ROC AUC
        if roc_auc > best_score:
            best_score = roc_auc
            best_model_name = model_name
            best_model = model
            best_run_id = info["run_id"]
            best_model_path = info["model_path"]

    logger.info(f"Best model on test set: {best_model_name} (ROC AUC={best_score:.4f})")

    # Register and promote best model in MLflow Model Registry
    client = MlflowClient()
    model_uri = f"runs:/{best_run_id}/model"
    result = mlflow.register_model(model_uri, "Churn_Detection_Model")
    client.transition_model_version_stage(
        name="Churn_Detection_Model",
        version=result.version,
        stage="Production",
        archive_existing_versions=True
    )
    logger.info(f"Registered and promoted {best_model_name} to Production in MLflow Model Registry.")

    # Save best model locally (overwrite for deployment)
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_model, os.path.join("model", "best_model.pkl"))

    return {
        "best_model_name": best_model_name,
        "best_model": best_model,
        "best_score": best_score,
        "best_run_id": best_run_id,
        "best_model_path": best_model_path
    }
