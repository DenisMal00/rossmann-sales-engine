"""
Main pipeline for training and evaluation.
Manually logs metrics and parameters to MLflow.
Promotes the model to production only if MAPE improves.
"""
import os
import mlflow
import mlflow.keras
import joblib
import logging
import warnings
from mlflow.tracking import MlflowClient
from datetime import datetime

CHAMPION_MAPE_BENCHMARK = 0.0855

# Silence system logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['GIT_PYTHON_REFRESH'] = 'quiet'
warnings.filterwarnings("ignore")
logging.getLogger("mlflow").setLevel(logging.ERROR)

from database import get_training_data
from train import train_global_model
from evaluate import evaluate_global

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT_NAME = "Rossmann_Sales_Forecasting"

def run_pipeline():
    # Setup MLflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    # Centralized hyperparameters
    params = {
        "window_size": 7,
        "batch_size": 256,
        "epochs": 50,
        "lstm_units": 58
    }

    print("Fetching data from database")
    df = get_training_data()
    if df is None or df.empty:
        print("Error: No data found.")
        return

    # Dynamic run name to distinguish experiments
    timestamp = datetime.now().strftime('%H:%M')
    run_name = f"LSTM_U{params['lstm_units']}_B{params['batch_size']}_{timestamp}"

    with mlflow.start_run(run_name=run_name):
        execute_training_workflow(params, df, client)


def execute_training_workflow(params, df, client):
    mlflow.log_params(params)

    print("Starting training...")
    model, scaler, history = train_global_model(df, params["batch_size"], params["epochs"],params["lstm_units"])

    # Auto-extract architecture from model layers
    for layer in model.layers:
        if 'lstm' in layer.name.lower():
            mlflow.log_param("lstm_units", getattr(layer, 'units', None))
        if 'dropout' in layer.name.lower():
            mlflow.log_param("dropout_rate", getattr(layer, 'rate', None))

    # Record training history metrics
    for epoch in range(len(history.history['loss'])):
        mlflow.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
        mlflow.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)

    print("Starting evaluation...")
    mae, mape, rmspe = evaluate_global(model, df, scaler)

    # Check if this run is better than our current best
    print("Checking historical performance...")
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="metrics.test_mape > 0",
        order_by=["metrics.test_mape ASC"],
        max_results=1
    )

    best_past_mape = CHAMPION_MAPE_BENCHMARK

    if runs:
        mlflow_best_mape = runs[0].data.metrics.get("test_mape")
        if mlflow_best_mape < CHAMPION_MAPE_BENCHMARK:
            best_past_mape = mlflow_best_mape
            print(f"MLflow record found: {best_past_mape:.4f}")
        else:
            print(f"MLflow records found but none beat the 8.55% benchmark.")
    else:
        print(f"No previous runs in MLflow. Using 8.55% benchmark.")

    is_new_best = False
    if mape < best_past_mape:
        is_new_best = True
        print(f"Congratulations! New record: {mape:.4f} (Previous best: {best_past_mape:.4f})")
    else:
        print(f"Current MAPE {mape:.4f} vs Best {best_past_mape:.4f}. Not updating the weights in /models.")

    mlflow.log_metric("test_mape", mape)
    mlflow.log_metric("test_mae", mae)
    mlflow.log_metric("test_rmspe", rmspe)

    # Log artifacts to MLflow
    scaler_temp = "scaler.joblib"
    joblib.dump(scaler, scaler_temp)
    mlflow.log_artifact(scaler_temp, artifact_path="model")
    mlflow.keras.log_model(model, artifact_path="model")

    # Update local production models if performance improved
    if is_new_best:
        print(f"Promoting model to {MODELS_DIR}")
        os.makedirs(MODELS_DIR, exist_ok=True)
        model.save(os.path.join(MODELS_DIR, 'sales_model.keras'))
        joblib.dump(scaler, os.path.join(MODELS_DIR, 'scaler.joblib'))

    if os.path.exists(scaler_temp):
        os.remove(scaler_temp)

    print("Run finished.")

if __name__ == "__main__":
    run_pipeline()